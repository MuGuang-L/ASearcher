#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn


def create_app(trace_dir: str, static_dir: str) -> FastAPI:
    app = FastAPI(title="ASearcher Light Trace Viewer", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    trace_root = Path(trace_dir).resolve()
    ui_root = Path(static_dir).resolve()

    def list_trace_files():
        if not trace_root.exists():
            return []
        return sorted(trace_root.rglob("*.trace.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    @app.get("/")
    async def root():
        return FileResponse(ui_root / "light_trace_viewer.html")

    @app.get("/light_trace_viewer.js")
    async def js():
        return FileResponse(ui_root / "light_trace_viewer.js")

    @app.get("/light_trace_viewer.css")
    async def css():
        return FileResponse(ui_root / "light_trace_viewer.css")

    @app.get("/api/episodes")
    async def episodes():
        results = []
        for path in list_trace_files():
            try:
                payload = json.loads(path.read_text())
            except Exception:
                continue
            results.append(
                dict(
                    qid=payload.get("qid", path.stem.replace(".trace", "")),
                    status=payload.get("status", "unknown"),
                    version=str(payload.get("version", path.parent.name)),
                    question=payload.get("question"),
                    reason=payload.get("reason"),
                    raw_scores=payload.get("raw_scores"),
                    normalized_scores=payload.get("normalized_scores"),
                    path=str(path),
                )
            )
        return {"episodes": results}

    @app.get("/api/episode")
    async def episode(path: str):
        file_path = Path(path).resolve()
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Trace file not found")
        try:
            file_path.relative_to(trace_root)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Path outside trace root") from exc
        return json.loads(file_path.read_text())

    return app


def main():
    parser = argparse.ArgumentParser(description="Serve ASearcher light trace viewer")
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    current_dir = Path(__file__).resolve().parent
    app = create_app(args.trace_dir, str(current_dir))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
