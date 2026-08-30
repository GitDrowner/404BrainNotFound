#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the 773086 calibrated demo and local model API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    os.environ["AIGC_DEVICE"] = args.device
    uvicorn.run(
        "aigc_detector.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
