"""Package-relative locations of the shipped QA configs and baseline artifacts."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PACKAGE_DIR / "configs"
BASELINES_DIR = PACKAGE_DIR / "baselines"

DEFAULT_CONFIG_NAME = "qa_config.yaml"
CAL_CONFIG_NAME = "qa_config_cal.yaml"
SCIENCE_CONFIG_NAME = "qa_config_science.yaml"
