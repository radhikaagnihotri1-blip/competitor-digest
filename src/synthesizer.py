import re
from anthropic import AsyncAnthropic
from . import config


async def synthesize(results: list[dict]) -> str:
    context = ""
    for r in results:
        context += f"\n## {r['company']}\n{r['summary']}\n"

    client = AsyncAnthropic()
    message = await client.messages.create(
        model=config.SYNTHESIS_MODEL,
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"""You are a senior competitive intelligence analyst writing a digest for a product manager.

Below are research summaries for {len(results)} {'company' if len(results) == 1 else 'companies'}, covering the last {config.SEARCH_DAYS} days.

{context}

Write a competitive intelligence digest as clean HTML. Do NOT include <!DOCTYPE>, <html>, <head>, or <body> tags — only the content HTML. Do NOT add any style, class, or id attributes to any tags. Use exactly these two sections:

<h2>Executive Summary</h2>
Write 3–5 sentences. Answer: what is the single most important thing happening across this competitive landscape right now? What patterns or trends cut across multiple companies? What should a PM pay attention to?

<h2>Competitive Digest</h2>
Write flowing paragraphs organized by theme, not by company. Group related moves across companies together. Compare strategies where relevant. Highlight what is most surprising or notable. This should read like a real CI briefing — insightful and opinionated, not a bullet dump.

Use <p> for paragraphs, <strong> for emphasis, <ul>/<li> for any lists. Keep it under 500 words total.""",
        }],
    )

    return message.content[0].text


def to_plain_text(html: str) -> str:
    text = re.sub(r"<h2>(.*?)</h2>", r"\n\1\n" + "-" * 40, html)
    text = re.sub(r"<li>(.*?)</li>", r"  • \1", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()
