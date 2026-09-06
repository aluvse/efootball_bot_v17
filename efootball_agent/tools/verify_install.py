from __future__ import annotations

import importlib
from pathlib import Path
import shutil
import sys


PACKAGE_IMPORTS = {
    "numpy": "numpy",
    "opencv-python": "cv2",
    "scipy": "scipy",
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "timm": "timm",
    "pywin32": "win32gui",
    "dxcam": "dxcam",
    "vgamepad": "vgamepad",
    "pytesseract": "pytesseract",
}


def _version(name, module_name):
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, "__version__", "OK")
    except Exception:
        return None


def find_tesseract():
    candidates = []
    which = shutil.which("tesseract")
    if which:
        candidates.append(Path(which))
    import os
    env = os.environ.get("TESSERACT_CMD") or os.environ.get("TESSERACT_PATH")
    if env:
        candidates.append(Path(env))
    candidates.extend([
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
    ])
    for path in candidates:
        try:
            if path.is_file():
                return str(path)
        except OSError:
            pass
    return None


def main():
    exit_code = 0
    print(f"python: {sys.version.split()[0]}")
    py_ok = sys.version_info[:2] == (3, 11)
    print(f"python 3.11 contract: {'OK' if py_ok else 'CHECK'}")
    if not py_ok:
        exit_code = 1
    print("[PYTHON PACKAGES]")
    versions = {}
    for name, module_name in PACKAGE_IMPORTS.items():
        version = _version(name, module_name)
        status = "OK" if version is not None else "MISSING"
        print(f"{name}: {status}" + (f" ({version})" if version else ""))
        versions[name] = version
        if version is None:
            exit_code = 1

    print("[EXTERNAL EXECUTABLES]")
    tesseract = find_tesseract()
    print(f"tesseract.exe: {'OK' if tesseract else 'MISSING'}" + (f" ({tesseract})" if tesseract else ""))
    if not tesseract:
        exit_code = 1

    print("[COMPATIBILITY]")
    try:
        import torch, torchvision
        expected = {"torch": "2.5.1", "torchvision": "0.20.1"}
        pair_ok = torch.__version__.split("+")[0] == expected["torch"] and torchvision.__version__.split("+")[0] == expected["torchvision"]
        print(f"torch/torchvision pair: {'OK' if pair_ok else 'CHECK'} ({torch.__version__}, {torchvision.__version__})")
        if not pair_ok:
            exit_code = 1
    except Exception as exc:
        print(f"torch/torchvision pair: ERROR ({type(exc).__name__}: {exc})")
        exit_code = 1

    print("[ACCELERATION]")
    try:
        import torch
        cpu_threads = 8
        cpu_interop = 1
        try:
            torch.set_num_threads(cpu_threads)
        except RuntimeError:
            pass
        try:
            torch.set_num_interop_threads(cpu_interop)
        except RuntimeError:
            pass
        print(f"torch device contract: CPU-only ({torch.__version__})")
        print(f"torch num threads: {torch.get_num_threads()} (target={cpu_threads})")
        print(f"torch interop threads: {torch.get_num_interop_threads()} (target={cpu_interop})")
    except Exception as exc:
        print(f"CPU probe: ERROR ({type(exc).__name__}: {exc})")
        exit_code = 1

    tf_ok = versions.get("transformers", "").split(".")[:3] == ["4", "57", "6"] if versions.get("transformers") else False
    print(f"transformers 4.57.6 contract: {'OK' if tf_ok else 'CHECK'}")
    if not tf_ok:
        exit_code = 1

    model = Path("models/rtdetr_r50vd")
    weights = list(model.glob("*.safetensors")) + list(model.glob("*.bin"))
    processor_files = (model / "preprocessor_config.json").exists() or (model / "image_processor_config.json").exists()
    print("[MODEL ASSETS]")
    print(f"rtdetr_dir: {'OK' if model.exists() else 'MISSING'} ({model})")
    print(f"rtdetr_config: {'OK' if (model / 'config.json').exists() else 'MISSING'}")
    print(f"rtdetr_weights: {'OK' if weights else 'MISSING'}")
    print(f"rtdetr_processor_config: {'OK' if processor_files else 'MISSING'}")
    if not model.exists() or not (model / "config.json").exists() or not weights or not processor_files:
        exit_code = 1
    print(f"verification: {'PASS' if exit_code == 0 else 'FAIL'}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
