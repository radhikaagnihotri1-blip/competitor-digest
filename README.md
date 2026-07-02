# Competitor Digest

A multi-agent tool that researches competitors in parallel and prints a structured digest to the console.

## Setup

**1. Get API keys (both free tiers)**
- Anthropic: https://console.anthropic.com → API Keys
- Tavily: https://tavily.com → Sign up → Dashboard → API Key

**2. Add your keys to `.env`**
```
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

```bash
# Pass companies as an argument
python run.py "Notion, Linear, Figma"

# Or run interactively
python run.py
```

## Project Structure

```
competitor-digest/
├── run.py              # Entry point
└── src/
    ├── config.py       # Model, search settings
    ├── orchestrator.py # Splits input, runs agents in parallel
    └── researcher.py   # One agent per company: search + Claude summary
```

## Roadmap

- [x] Phase 1: CLI → parallel research agents → console output
- [ ] Phase 2: Synthesis agent + Resend email digest
- [ ] Phase 3: Web frontend (portfolio-ready)
