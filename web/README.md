# Dashboard (React + TypeScript)

Reads everything from the FastAPI service (`src/riskengine/api.py`).

```bash
uv run uvicorn riskengine.api:app --port 8000   # from the repo root
cd web && npm install && npm run dev            # http://localhost:5173
```

In development, Vite forwards `/api/*` to `localhost:8000` (`vite.config.ts`).

- `src/api.ts`: fetch helper and `useApi` hook
- `src/types.ts`: shapes of the API responses
- `src/pages/`: one file per page; `src/components/`: shared parts
- Light/dark theme: follows the OS until the toggle is used, then remembered in the browser.
