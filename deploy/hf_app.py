"""Entry point when the app runs as a Hugging Face Space (copied there as app.py).

Free Spaces cannot run Docker, so the Space is declared as a Gradio app and this file starts our own
server instead: one process serving the dashboard at / and the API under /api on port 7860.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import uvicorn  # noqa: E402

from riskengine.site import app  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
