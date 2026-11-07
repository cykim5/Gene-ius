"""Point every test at the tiny fixture graph in tests/fixtures."""
import os
from pathlib import Path

os.environ["GENEIUS_DATA_DIR"] = str(Path(__file__).parent / "fixtures")
os.environ["GENEIUS_USE_STUBS"] = "0"
# Keep tests independent of local, untracked files in data/processed.
os.environ["GENEIUS_FEATURE_TABLE"] = str(Path(__file__).parent / "fixtures" / "no_feature_table.csv")
os.environ["GENEIUS_BASELINE_PREDICTIONS"] = str(Path(__file__).parent / "fixtures" / "no_predictions.csv")
os.environ["GENEIUS_PROCESSED_DIR"] = str(Path(__file__).parent / "fixtures" / "processed")
os.environ["GENEIUS_VARIANTS_CSV"] = str(Path(__file__).parent / "fixtures" / "processed" / "variants.csv")
