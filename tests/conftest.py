import importlib.util
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
LOW_LEVEL = REPO.parent / "low_level_controller"  # sibling repo; present on the dev laptop, usually not on the server


def load_script(name: str):
    """Import scripts/<name>.py as a module (skips the test if one of its imports is missing in this env)."""
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as e:
        pytest.skip(f"scripts/{name}.py needs {e.name}")
    return module
