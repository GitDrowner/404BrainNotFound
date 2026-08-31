#!/usr/bin/env python3
"""Run a Python module/script after appending a fallback site-packages directory.

The selected interpreter's own packages remain first, so its matched torch/CUDA stack wins.
Only packages absent from that environment (for example Pillow or scikit-learn) fall back to
the project environment. This file performs no installation and changes no environment.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    fallback = Path(os.environ["AIGC_FALLBACK_SITE_PACKAGES"]).resolve()
    if not fallback.is_dir():
        raise RuntimeError(f"Fallback site-packages is unavailable: {fallback}")
    sys.path.append(str(fallback))
    command = sys.argv[1:]
    if not command:
        raise RuntimeError("Expected '-m MODULE [ARGS]' or 'SCRIPT [ARGS]'")
    if command[0] == "-m":
        if len(command) < 2:
            raise RuntimeError("-m requires a module name")
        module = command[1]
        sys.argv = [module, *command[2:]]
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    else:
        script = command[0]
        sys.argv = [script, *command[1:]]
        runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
