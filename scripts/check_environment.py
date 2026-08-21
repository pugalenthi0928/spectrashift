from __future__ import annotations

import importlib.util
import json
import os
import platform
import sys


MODULES = ("numpy", "torch", "sklearn", "yaml", "rasterio", "xarray", "netCDF4")


def main() -> None:
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "modules": {name: importlib.util.find_spec(name) is not None for name in MODULES},
        "environment_flags": {
            "HYPERSIGMA_CHECKPOINT": bool(os.getenv("HYPERSIGMA_CHECKPOINT")),
            "HYPERSIGMA_SOURCE": bool(os.getenv("HYPERSIGMA_SOURCE")),
            "HYPERFREE_CHECKPOINT": bool(os.getenv("HYPERFREE_CHECKPOINT")),
            "HYPERFREE_SOURCE": bool(os.getenv("HYPERFREE_SOURCE")),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

