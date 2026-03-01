#!/usr/bin/env bash

set -euo pipefail

# One-command local CI parity run
python -m pip install --upgrade pip
python -m pip install tox
python -m tox
