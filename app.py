"""Style Search demo for Madam. Run:  python app.py  then open http://localhost:8000"""

from pathlib import Path

import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse

from search import StyleSearch

app = FastAPI(title="Madam Style Search Demo")

print("Building search index (first run downloads the embedding model, ~130MB)...")
engine = StyleSearch()
print(f"Ready. Backend: {engine.backend.name}")


@app.get("/")
def home():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/search")
def search(q: str = Query(..., min_length=2), k: int = 8):
    return engine.search(q, k=k)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
