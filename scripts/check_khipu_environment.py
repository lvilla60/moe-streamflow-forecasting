"""Print Khipu runtime, dependency, CUDA, and project-file diagnostics."""

import importlib
import os
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def import_version(module_name, display_name=None):
    display_name = display_name or module_name
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        print(f"{display_name} version: MISSING ({error})")
        return None
    print(f"{display_name} version: {getattr(module, '__version__', 'unknown')}")
    return module


def main():
    print(f"Python version: {sys.version.replace(os.linesep, ' ')}")
    print(f"Platform: {platform.platform()}")

    torch = import_version("torch")
    if torch is not None:
        print(f"Torch CUDA runtime: {torch.version.cuda}")
        available = bool(torch.cuda.is_available())
        print(f"torch.cuda.is_available(): {available}")
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        if available:
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"Current CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        else:
            print("Current CUDA device name: unavailable")

    h5py = import_version("h5py")
    import_version("numpy")
    import_version("pandas")
    import_version("sklearn", "scikit-learn")
    import_version("matplotlib")

    print(f"Current working directory: {Path.cwd()}")
    for relative_path in (
        "Input/train-001.h5", "Input/test.h5", "outputs/train_stats.json"
    ):
        path = ROOT / relative_path
        print(f"{relative_path}: {'FOUND' if path.is_file() else 'MISSING'}")

    if torch is None or h5py is None:
        print("Critical environment check failed: torch and h5py are required.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
