#!/bin/bash
cd "$(dirname "$0")" || exit 1
source "./env.sh"
python3 cb0tokens2tokens.py "$@"
