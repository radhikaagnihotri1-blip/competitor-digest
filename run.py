import asyncio
import sys
from dotenv import load_dotenv
from src.orchestrator import parse_companies, run

load_dotenv()


def main() -> None:
    if len(sys.argv) > 1:
        input_str = " ".join(sys.argv[1:])
    else:
        input_str = input("Enter company names (comma-separated): ").strip()

    companies = parse_companies(input_str)
    if not companies:
        print("No companies provided. Exiting.")
        return

    asyncio.run(run(companies))


if __name__ == "__main__":
    main()
