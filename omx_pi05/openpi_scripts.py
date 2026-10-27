"""Load openpi's top-level scripts (scripts/*.py are not importable modules) so our wrappers can call their main()."""

import importlib.util
import pathlib
import sys
import types

OPENPI_SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "third_party" / "openpi" / "scripts"


def load(name: str) -> types.ModuleType:
    """load("train") → module object of third_party/openpi/scripts/train.py."""
    path = OPENPI_SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"openpi_scripts_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
