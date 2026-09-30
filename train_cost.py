"""Compatibility entry point: all cost training now uses the lifecycle gates."""
from pipeline import run
import json
if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
