#!/bin/bash
args=()
for arg in "$@"; do
    if [[ -e "$arg" ]]; then
        args+=("$(realpath "$arg")")
    else
        args+=("$arg")
    fi
done
cd "$(dirname "$0")" || exit 1
source "./env.sh"
python3 jsonl2conditions.py "${args[@]}"
