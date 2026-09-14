"""Tools for the Medical Agent."""

from .classification import *
from .report_generation import *
from .segmentation import *
from .xray_vqa import *
from .llava_med import *
from .grounding import *
# PATCH: ChestXRayGeneratorTool (generation.py) imports diffusers, which in recent
# versions requires a newer transformers than the commit pinned in pyproject.toml
# (fails with: cannot import name 'Dinov2WithRegistersConfig'). The tool also needs
# RoentGen weights that are not publicly downloadable, so treat it as optional
# rather than letting it break the import of every other tool. Not a plain
# ImportError: diffusers re-raises lazy-import failures as RuntimeError.
try:
    from .generation import *
except Exception as _generation_import_error:  # pragma: no cover
    # print, not warnings.warn: something in the import chain installs a global
    # "ignore" filter, which swallows the warning before the user ever sees it.
    import sys as _sys

    print(
        f"[medrax] ChestXRayGeneratorTool unavailable: {_generation_import_error}",
        file=_sys.stderr,
    )
from .dicom import *
from .utils import *
