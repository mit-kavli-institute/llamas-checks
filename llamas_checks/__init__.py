"""llamas_checks: raw-frame QA "image checks" for LLAMAS multi-extension FITS files.

Standalone port of the QA framework from the llamas-pyjamas reduction pipeline.
It evaluates YAML-driven rule sets (bias/overscan levels, noise, saturation,
header sanity, structural completeness) against raw CAL and SCI frames and
returns PASS/WARN/FAIL verdicts with JSON reports.

Entry points:
    llamas-checks           (llamas_checks.QA_assess:main)         observing-GUI gate
    llamas-checks-engine    (llamas_checks.qa_engine:main)         YAML QA engine
    llamas-checks-validate  (llamas_checks.qa_config_validator:main)
    python -m llamas_checks <file>

The public names are re-exported lazily (PEP 562) so that
``python -m llamas_checks.qa_engine`` does not import the submodule twice and
emit a RuntimeWarning -- in the parent and in every spawned batch worker.
"""

import importlib

__version__ = "0.1.0"

__all__ = ["QAEngine", "QAEngineError", "check_image", "__version__"]

_LAZY_EXPORTS = {
    "QAEngine": ".qa_engine",
    "QAEngineError": ".qa_engine",
    "check_image": ".llamasQATests",
}


def __getattr__(name):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name, __name__), name)
    globals()[name] = value
    return value
