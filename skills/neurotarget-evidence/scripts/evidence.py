from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from neurotarget.scienceclaw_tools import main

if __name__ == "__main__":
    main("evidence")
