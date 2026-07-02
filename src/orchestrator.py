import asyncio
from . import researcher, synthesizer, mailer, config


def parse_companies(input_str: str) -> list[str]:
    return [c.strip() for c in input_str.split(",") if c.strip()]


def print_result(result: dict) -> None:
    bar = "=" * 52
    print(f"\n{bar}")
    print(f"  {result['company'].upper()}")
    print(bar)
    print(result["summary"])
    if result["sources"]:
        print("\nSources:")
        for url in result["sources"]:
            print(f"  {url}")


async def run(companies: list[str]) -> None:
    noun = "company" if len(companies) == 1 else "companies"
    print(f"\nResearching {len(companies)} {noun} in parallel...")

    results = await asyncio.gather(*[researcher.research(c) for c in companies])

    for result in results:
        print_result(result)

    print(f"\n{'=' * 52}")
    print("  SYNTHESIZING DIGEST")
    print(f"{'=' * 52}")
    digest_html = await synthesizer.synthesize(results)
    print(synthesizer.to_plain_text(digest_html))

    print(f"\n{'=' * 52}")
    print("  SENDING EMAIL")
    print(f"{'=' * 52}")
    mailer.send(digest_html, companies, list(results))
    print(f"Digest sent to {config.TO_EMAIL}")

    print(f"\n{'=' * 52}\nDone.\n")
