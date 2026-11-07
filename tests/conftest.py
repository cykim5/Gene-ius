"""Point every test at the tiny fixture graph in tests/fixtures."""
import os
from pathlib import Path

os.environ["GENEIUS_DATA_DIR"] = str(Path(__file__).parent / "fixtures")
os.environ["GENEIUS_USE_STUBS"] = "0"
os.environ["GENEIUS_RANKER"] = "overlap"
