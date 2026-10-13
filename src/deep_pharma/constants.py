from pathlib import Path

COMPETITION_SLUG = "deep-pharma-multimodal-intelligence-challenge-data-hack-3-0"

ID_COLUMN = "Pair_ID"
SEVERITY_COLUMN = "Severity"
BINARY_PREFIX = "Target_Binary_"
PRR_PREFIX = "Target_PRR_"

EXPECTED_BINARY_TARGETS = 50
EXPECTED_PRR_TARGETS = 50
EXPECTED_SUBMISSION_COLUMNS = 102
ALLOWED_SEVERITY_CLASSES = frozenset({"Major", "Moderate", "Minor", "Contraindication"})

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_REPORT_PATH = Path("reports/data_contract.json")
