import asyncio
import json
import os
import re

import httpx
from anthropic import AsyncAnthropic
from tavily import AsyncTavilyClient

from . import config


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

async def _noop():
    return None


def _safe(v):
    """Coerce asyncio.gather exceptions to None."""
    return None if isinstance(v, Exception) else v


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text.strip())
    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Part 1: Classification
# ──────────────────────────────────────────────────────────────────────────────

async def _classify(company: str) -> dict:
    client = AsyncAnthropic()
    msg = await client.messages.create(
        model=config.MODEL,
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": (
                f'Classify the company "{company}". '
                "Return ONLY valid JSON, no markdown:\n"
                "{\n"
                '  "industry": "developer-tools|design-tools|api-first|consumer-app|enterprise-saas|fintech|ecommerce|media|other",\n'
                '  "is_developer_facing": true,\n'
                '  "is_venture_backed": true,\n'
                '  "has_mobile_app": true,\n'
                '  "is_public_company": false,\n'
                '  "has_public_github": true\n'
                "}"
            ),
        }],
    )
    raw = _strip_fences(msg.content[0].text)
    try:
        return json.loads(raw)
    except Exception:
        return {
            "industry": "other",
            "is_developer_facing": False,
            "is_venture_backed": False,
            "has_mobile_app": False,
            "is_public_company": False,
            "has_public_github": False,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Part 2: External data fetchers
# ──────────────────────────────────────────────────────────────────────────────

# Terms that confirm a search snippet is about a tech company
_COMPANY_SNIPPET_TERMS = {"company", "software", "startup", "platform", "app", "saas",
                           "corporation", "inc", "llc", "ltd", "technology", "technologies"}

# Terms that must appear in the extract to pass validation
_COMPANY_EXTRACT_TERMS = {"founded", "inc.", "inc,", "software", "app", "platform",
                           "ceo", "startup", "venture", "series", "billion", "million",
                           "employees", "headquartered", "acquired", "subscription", "productivity"}

# Terms that immediately disqualify an extract (wrong domain entirely)
_BLACKLIST_TERMS = {"semiconductor", "analog", "integrated circuit", "transistor",
                    "non-linear editing", "nle", "video editing", "film editing",
                    "mathematics", "algebra", "calculus", "physics", "particle",
                    "genus ", "species ", "biology", "chemistry", "electrode"}

# Map classification industry → human-readable search qualifier
_INDUSTRY_SEARCH_TERM = {
    "developer-tools":  "project management developer software",
    "design-tools":     "design software",
    "api-first":        "API software company",
    "consumer-app":     "app company",
    "enterprise-saas":  "enterprise software company",
    "fintech":          "fintech financial technology company",
    "ecommerce":        "ecommerce software company",
    "media":            "media platform company",
    "other":            "software company",
}


def _is_company_extract(extract: str) -> bool:
    text = extract.lower()
    if any(term in text for term in _BLACKLIST_TERMS):
        return False
    return any(term in text for term in _COMPANY_EXTRACT_TERMS)


def _parse_extract_funding(extract: str) -> str | None:
    """Scan plain-text Wikipedia summary for any dollar amount with a unit."""
    if not extract:
        return None
    m = re.search(r'\$[\d,.]+\s*(?:billion|million|[BMK])\b', extract, re.IGNORECASE)
    if m:
        return m.group(0).strip()[:60]
    return None


def _parse_wikitext_funding(wikitext: str) -> str | None:
    """Extract a funding amount string from Wikipedia infobox wikitext."""
    # 1. Infobox field patterns
    for pattern in [
        r'\|\s*(?:total_)?funding(?:_round)?\s*=\s*([^\|\n\}]{1,80})',
        r'\|\s*raised\s*=\s*([^\|\n\}]{1,80})',
    ]:
        m = re.search(pattern, wikitext, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            val = re.sub(r'\[\[([^\]|]*\|)?([^\]]*)\]\]', r'\2', val)  # [[link|text]] → text
            val = re.sub(r'\{\{[^}]*\}\}', '', val).strip()             # remove templates
            if val and re.search(r'[\$\d]', val) and re.search(r'(?:million|billion|[MB])', val, re.I):
                return val[:60]

    # 2. Proximity: dollar amount near "funding" or "raised" within 100 chars
    m = re.search(
        r'(?:fund(?:ing|ed)|raised?)\D{0,80}?(\$[\d,.]+\s*(?:million|billion|[MBK]))',
        wikitext, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()

    return None


async def _fetch_wikipedia(company: str, industry: str = "other") -> dict | None:
    """Fetch founding year / HQ from Wikipedia + Wikidata. Uses industry-aware search to avoid
    returning wrong articles for ambiguous names (e.g. Linear → semiconductor company)."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            headers = {"User-Agent": "CompetitorDigest/1.0 (research tool)"}
            industry_term = _INDUSTRY_SEARCH_TERM.get(industry, "software company")

            # Step 1: run search queries in priority order, take first snippet hit
            page_title = None
            for search_query in [
                f"{company} {industry_term}",
                f"{company} company",
                f"{company} software",
                f"{company} startup",
            ]:
                sr = await client.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "list": "search",
                        "srsearch": search_query,
                        "srlimit": 5,
                        "format": "json",
                    },
                    headers=headers,
                )
                if sr.status_code != 200:
                    continue
                for result in sr.json().get("query", {}).get("search", []):
                    snippet = (result.get("snippet") or "").lower()
                    if any(term in snippet for term in _COMPANY_SNIPPET_TERMS):
                        page_title = result.get("title", "")
                        break
                if page_title:
                    break

            # Step 2: resolve candidates → REST summary, validate extract
            candidates = []
            if page_title:
                candidates.append(page_title)
            candidates.append(f"{company} (company)")
            candidates.append(company)

            data = None
            for title in candidates:
                r = await client.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}",
                    headers=headers,
                )
                if r.status_code != 200:
                    continue
                candidate_data = r.json()
                if candidate_data.get("type") == "disambiguation":
                    continue
                extract = candidate_data.get("extract") or ""
                if not _is_company_extract(extract):
                    continue
                data = candidate_data
                break

            if not data:
                return None

            description = data.get("extract", "")[:400]
            wikidata_id = data.get("wikibase_item")  # e.g. "Q19841877"

            founding_year = None
            hq = None

            if wikidata_id:
                # Wikidata entity fetch for P571 (inception) and P159 (HQ)
                wd = await client.get(
                    "https://www.wikidata.org/w/api.php",
                    params={
                        "action": "wbgetentities",
                        "ids": wikidata_id,
                        "props": "claims",
                        "format": "json",
                    },
                )
                if wd.status_code == 200:
                    claims = wd.json().get("entities", {}).get(wikidata_id, {}).get("claims", {})

                    # P571 = inception date
                    p571 = claims.get("P571", [])
                    if p571:
                        time_val = (
                            p571[0]
                            .get("mainsnak", {})
                            .get("datavalue", {})
                            .get("value", {})
                            .get("time", "")
                        )
                        if time_val:
                            founding_year = time_val[1:5]  # "+YYYY-MM-DDT..."

                    # P159 = headquarters location — just grab the label via entity ID
                    p159 = claims.get("P159", [])
                    if p159:
                        hq_id = (
                            p159[0]
                            .get("mainsnak", {})
                            .get("datavalue", {})
                            .get("value", {})
                            .get("id")
                        )
                        if hq_id:
                            hq_r = await client.get(
                                "https://www.wikidata.org/w/api.php",
                                params={
                                    "action": "wbgetentities",
                                    "ids": hq_id,
                                    "props": "labels",
                                    "languages": "en",
                                    "format": "json",
                                },
                            )
                            if hq_r.status_code == 200:
                                hq = (
                                    hq_r.json()
                                    .get("entities", {})
                                    .get(hq_id, {})
                                    .get("labels", {})
                                    .get("en", {})
                                    .get("value")
                                )

            # Step 3: fetch wikitext (section 0) for infobox funding data
            funding_string = None
            wikitext_title = data.get("title", "")
            if wikitext_title:
                wt = await client.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "prop": "revisions",
                        "rvprop": "content",
                        "titles": wikitext_title,
                        "format": "json",
                        "formatversion": "2",
                        "rvsection": "0",
                    },
                    headers=headers,
                )
                if wt.status_code == 200:
                    pages = wt.json().get("query", {}).get("pages", [])
                    if pages:
                        revisions = pages[0].get("revisions", [])
                        if revisions:
                            wikitext = revisions[0].get("content", "")
                            funding_string = _parse_wikitext_funding(wikitext)

            # Fallback: scan plain-text Wikipedia summary for any funding figure
            if not funding_string:
                funding_string = _parse_extract_funding(description)

            return {
                "description": description,
                "founding_year": founding_year,
                "hq": hq,
                "founders": None,
                "funding_string": funding_string,
            }
    except Exception:
        return None


async def _fetch_trends(company: str) -> dict | None:
    """Google Trends via pytrends (synchronous, run in executor). 30s timeout."""
    def _sync():
        try:
            from pytrends.request import TrendReq  # imported here to avoid import-time side effects
            pt = TrendReq(hl="en-US", tz=0, timeout=(8, 20), retries=1, backoff_factor=0.5)
            pt.build_payload([company], timeframe="today 5-y", gprop="")
            df = pt.interest_over_time()
            if df is None or df.empty or company not in df.columns:
                return None
            return {
                "interest_over_time": [
                    {"date": str(idx.date()), "value": int(row[company])}
                    for idx, row in df.iterrows()
                    if not row.get("isPartial", False)
                ]
            }
        except Exception:
            return None

    try:
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(loop.run_in_executor(None, _sync), timeout=30.0)
    except Exception:
        return None


async def _fetch_app_store(company: str) -> dict | None:
    """iTunes Search API — free, no key needed."""
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(
                "https://itunes.apple.com/search",
                params={"term": company, "entity": "software", "limit": 5},
            )
            if r.status_code != 200:
                return None
            results = r.json().get("results", [])
            if not results:
                return None
            app = results[0]
            return {
                "name": app.get("trackName"),
                "rating": app.get("averageUserRating"),
                "rating_count": app.get("userRatingCount"),
                "version": app.get("version"),
                "last_updated": (app.get("currentVersionReleaseDate") or "")[:10],
            }
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# CLI text formatter (unchanged — keeps CLI / email paths working)
# ──────────────────────────────────────────────────────────────────────────────

def _structured_to_text(s: dict, company: str) -> str:
    lines = []

    lines.append("PRICING CHANGES:")
    changes = s.get("pricing_changes", [])
    tiers   = s.get("pricing_tiers", [])
    if changes:
        for c in changes:
            if c.get("old_price") is not None and c.get("new_price") is not None:
                arrow = f"${c['old_price']} → ${c['new_price']}"
            elif c.get("new_price") is not None:
                arrow = f"now ${c['new_price']}/mo"
            else:
                arrow = c.get("note", "change noted")
            date_str = f" ({c['date']})" if c.get("date") else ""
            lines.append(f"  - {c.get('tier', 'Plan')}: {arrow}{date_str}")
    elif tiers:
        for t in tiers[:4]:
            price = "Free" if t.get("price") == 0 else f"${t.get('price', '?')}/mo"
            lines.append(f"  - {t.get('name', 'Tier')}: {price}")
    else:
        lines.append("  - None found")

    lines.append("\nPRODUCT LAUNCHES:")
    launches = s.get("product_launches", [])
    if launches:
        for launch in launches:
            date_str = f" ({launch['date']})" if launch.get("date") else ""
            desc = launch.get("description", "")
            lines.append(f"  - {launch.get('name', 'Feature')}{date_str}: {desc}")
    else:
        lines.append("  - None found")

    lines.append("\nNOTABLE NEWS:")
    news = s.get("notable_news", [])
    if news:
        for item in news:
            lines.append(f"  - {item.get('headline', '')}")
    else:
        lines.append("  - None found")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main research function
# ──────────────────────────────────────────────────────────────────────────────

async def research(company: str, on_progress=None) -> dict:
    # Step 1: classify
    if on_progress:
        await on_progress("Classifying company...")
    classification = await _classify(company)

    # Step 2: Tavily parallel search
    if on_progress:
        await on_progress("Searching news and pricing...")

    tavily = AsyncTavilyClient()
    general_queries = [
        f"{company} pricing changes",
        f"{company} product launch OR new feature",
        f"{company} news announcement",
    ]
    funding_queries = [
        f"{company} app funding history all rounds raised million",
        f"{company} software series A B seed funding investors million",
        f"{company} startup total funding raised history crunchbase",
    ]
    all_queries = general_queries + funding_queries
    search_results = await asyncio.gather(*[
        tavily.search(q, max_results=8 if i >= 3 else config.MAX_RESULTS_PER_SEARCH,
                      days=3650 if i >= 3 else config.SEARCH_DAYS)
        for i, q in enumerate(all_queries)
    ])

    # Combine and deduplicate the three funding queries by URL
    _seen_funding_urls: set = set()
    _combined_funding = []
    for _r in (search_results[3].get("results", [])
               + search_results[4].get("results", [])
               + search_results[5].get("results", [])):
        if _r.get("url") not in _seen_funding_urls:
            _seen_funding_urls.add(_r.get("url"))
            _combined_funding.append(_r)

    context = ""
    sources = []
    for query, result in zip(general_queries, search_results[:3]):
        context += f"\n### Search: {query}\n"
        for r in result.get("results", []):
            context += f"Title: {r['title']}\nURL: {r['url']}\nContent: {r['content']}\n\n"
            sources.append(r["url"])
    context += "\n### Search: funding rounds and investment\n"
    for r in _combined_funding:
        context += f"Title: {r['title']}\nURL: {r['url']}\nContent: {r['content']}\n\n"
        sources.append(r.get("url", ""))

    # Step 3: Claude extraction
    if on_progress:
        await on_progress("Analyzing with Claude...")

    client = AsyncAnthropic()
    message = await client.messages.create(
        model=config.MODEL,
        max_tokens=1600,
        messages=[{
            "role": "user",
            "content": f"""You are a competitive intelligence analyst. You are extracting data for {company}, a {classification.get("industry", "software")} company — ignore any search results about other companies that share a similar name (e.g. investors, funds, or unrelated businesses named "{company}"). Based only on the search results below, extract structured data about {company} in the last {config.SEARCH_DAYS} days. For funding specifically: extract ALL historical funding rounds you can find, not just recent ones. Include Seed, Series A, B, C, D and any other rounds mentioned. We want the complete funding history from founding to present.

{context}

Return ONLY a valid JSON object — no markdown, no explanation, just the JSON.
Use this exact schema:
{{
  "pricing_tiers": [
    {{"name": "Tier name", "price": 0, "billing": "monthly"}}
  ],
  "pricing_changes": [
    {{"tier": "Tier name", "old_price": 0, "new_price": 0, "date": "YYYY-MM", "note": "brief context"}}
  ],
  "product_launches": [
    {{"name": "Feature or product name", "date": "YYYY-MM", "description": "one sentence max 20 words"}}
  ],
  "notable_news": [
    {{"headline": "one sentence", "date": "YYYY-MM"}}
  ],
  "sentiment": {{
    "pricing_aggressiveness": 0,
    "ai_investment": 0,
    "growth_focus": 0,
    "enterprise_focus": 0
  }},
  "funding": {{
    "total_raised": null,
    "valuation": null,
    "rounds": [
      {{"year": "2019", "series": "Series A", "amount_m": 6.5}}
    ]
  }}
}}
Rules:
- sentiment values are integers -2 (very low) to +2 (very high)
- pricing_aggressiveness: -2=very cheap/generous, +2=very expensive/aggressive
- ai_investment: -2=no AI focus, +2=massive AI investment
- growth_focus: -2=consolidating, +2=aggressive growth mode
- enterprise_focus: -2=consumer/SMB only, +2=heavy enterprise push
- Include up to 5 items per list
- Use null for unknown dates
- price field is numeric dollars per month (0 for free tiers)
- funding.rounds: extract ALL rounds found, up to 12; if year is unknown omit the round entirely; if amount is unknown omit amount_m (the round can still appear without it).
- BE AGGRESSIVE on amounts: if you see any number followed by "million", "M", "billion", or "B" within 20 words of "funding", "raised", "round", "Series", "investment", or "secured" — extract it as amount_m. Do NOT require 100% certainty. "$82M Series C", "raised $82 million", "secured $82M" all → amount_m: 82.0.
- funding.total_raised: TOTAL venture capital RAISED (money investors gave the company), NOT valuation. String like "$52M" or "$343.2M". If a cumulative figure is mentioned ("has raised $343M to date") put it here. If not stated, sum round amounts. Never put valuation here.
- funding.valuation: the company's estimated worth or post-money valuation (e.g. "$1.25B"), NOT the amount raised. Null if not found.
- funding.rounds[].amount_m: amount raised in THAT SPECIFIC ROUND only, as a float in millions (e.g. 15.0 for $15M Series A, 82.0 for $82M Series C). NOT cumulative, NOT valuation. When in doubt, include it — a slightly wrong number is better than null.""",
        }],
    )

    raw = _strip_fences(message.content[0].text)
    try:
        structured = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        structured = {
            "pricing_tiers": [],
            "pricing_changes": [],
            "product_launches": [],
            "notable_news": [],
            "sentiment": {
                "pricing_aggressiveness": 0,
                "ai_investment": 0,
                "growth_focus": 0,
                "enterprise_focus": 0,
            },
            "funding": {"total_raised": None, "valuation": None, "rounds": []},
        }

    summary = _structured_to_text(structured, company)

    # Step 4: external signals (parallel, gated by classification)
    if on_progress:
        await on_progress("Fetching external signals...")

    has_app = classification.get("has_mobile_app", False)

    wikipedia, trends, app_store = await asyncio.gather(
        _fetch_wikipedia(company, industry=classification.get("industry", "other")),
        _fetch_trends(company),
        _fetch_app_store(company) if has_app else _noop(),
        return_exceptions=True,
    )

    # Deduplicate sources
    seen = set()
    unique_sources = []
    for url in sources:
        if url not in seen:
            seen.add(url)
            unique_sources.append(url)

    return {
        "company":        company,
        "summary":        summary,
        "sources":        unique_sources[:10],
        "structured":     structured,
        "classification": classification,
        "wikipedia":      _safe(wikipedia),
        "trends":         _safe(trends),
        "app_store":      _safe(app_store),
    }
