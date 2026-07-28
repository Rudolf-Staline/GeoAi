"""Verify the minimum project environment without accessing private data."""

from __future__ import annotations

import platform
import sys

import numpy
import pandas
import sklearn
import yaml


def main() -> None:
    print(f"python={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")
    print(f"numpy={numpy.__version__}")
    print(f"pandas={pandas.__version__}")
    print(f"scikit-learn={sklearn.__version__}")
    print(f"pyyaml={yaml.__version__}")


if __name__ == "__main__":
    main()
