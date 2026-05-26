import sys
from pathlib import Path

# Make the project root importable so tests can do `from pipeline_funcs import ...`
sys.path.insert(0, str(Path(__file__).parent.parent))
