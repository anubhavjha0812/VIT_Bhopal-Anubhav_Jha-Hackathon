"""One server for the whole demo: the API under /api and the built dashboard at /.

    cd web && npm run build && cd ..          # once, writes web/dist
    uv run uvicorn riskengine.site:app --port 8000   # http://localhost:8000

In development the dashboard runs on Vite instead (`npm run dev`) and calls the API app directly.
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from riskengine import api
from riskengine.paths import ROOT

DIST = ROOT / "web" / "dist"


@asynccontextmanager
async def lifespan(_: FastAPI):
    # A mounted app's own lifespan does not run, so start the model warm-up here.
    asyncio.get_running_loop().run_in_executor(None, api.warm_up)
    yield


app = FastAPI(title="AI Risk Engine", lifespan=lifespan, docs_url=None, redoc_url=None)
app.mount("/api", api.app)


@app.get("/{path:path}", include_in_schema=False)
def dashboard(path: str) -> FileResponse:
    """Built files as they are; any other path gets index.html so the React router handles it."""
    if not (DIST / "index.html").exists():
        raise HTTPException(
            503, "dashboard not built: run `cd web && npm install && npm run build`"
        )
    file = (DIST / path).resolve()
    if path and file.is_file() and file.is_relative_to(DIST.resolve()):
        return FileResponse(file)
    return FileResponse(DIST / "index.html")
