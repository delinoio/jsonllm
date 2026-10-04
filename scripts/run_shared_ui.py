"""Compatibility script for the documented registered-UI example."""

import sys

from jsonllm.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", *sys.argv[1:]]))
