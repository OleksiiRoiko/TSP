from typing import TYPE_CHECKING
import importlib

if TYPE_CHECKING:
    from . import cli

__all__ = ["cli"]
__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "cli":
        return importlib.import_module(f"{__name__}.cli")
    raise AttributeError(name)
