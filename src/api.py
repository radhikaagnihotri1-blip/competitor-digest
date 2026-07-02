import asyncio
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from . import researcher, synthesizer, mailer

app = FastAPI()

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


class RunRequest(BaseModel):
    companies: list[str]


class EmailRequest(BaseModel):
    html: str
    companies: list[str]
    sources_by_company: list[dict]


class ClassifyRequest(BaseModel):
    company: str


@app.post("/run")
async def run_endpoint(req: RunRequest):
    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()

        for company in req.companies:
            yield {"event": "agent_queued", "data": json.dumps({"company": company})}

        async def research_one(company: str) -> dict:
            await queue.put(("agent_start", {"company": company}))

            async def on_progress(msg: str) -> None:
                await queue.put(("agent_progress", {"company": company, "message": msg}))

            try:
                result = await researcher.research(company, on_progress=on_progress)
            except Exception as e:
                result = {"company": company, "summary": f"Research failed: {e}", "sources": []}

            await queue.put(("agent_done", result))
            return result

        tasks = [asyncio.create_task(research_one(c)) for c in req.companies]

        completed = 0
        while completed < len(req.companies):
            event_type, data = await queue.get()
            if event_type == "agent_done":
                completed += 1
            yield {"event": event_type, "data": json.dumps(data)}

        results = await asyncio.gather(*tasks)

        yield {"event": "synthesis_start", "data": json.dumps({})}
        try:
            digest_html = await synthesizer.synthesize(list(results))
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"message": str(e)})}
            return

        yield {"event": "synthesis_done", "data": json.dumps({"html": digest_html})}
        yield {"event": "complete", "data": json.dumps({})}

    return EventSourceResponse(event_stream())


@app.post("/classify")
async def classify_endpoint(req: ClassifyRequest):
    result = await researcher._classify(req.company)
    return result


@app.post("/email")
async def email_endpoint(req: EmailRequest):
    await asyncio.to_thread(mailer.send, req.html, req.companies, req.sources_by_company)
    return {"ok": True}


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")
