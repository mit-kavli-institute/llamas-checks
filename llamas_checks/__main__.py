"""Allow ``python -m llamas_checks <file>`` as an alias for ``llamas-checks``."""

from .QA_assess import main

if __name__ == "__main__":
    raise SystemExit(main())
