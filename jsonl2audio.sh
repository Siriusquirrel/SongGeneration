#!/bin/bash
args=()
jsonl_file=""
for arg in "$@"; do
    if [[ -e "$arg" ]]; then
        resolved_path="$(realpath "$arg")"
        args+=("$resolved_path")
        if [[ "$resolved_path" == *.jsonl ]]; then
            jsonl_file="$resolved_path"
        fi
    else
        args+=("$arg")
    fi
done
cd "$(dirname "$0")" || exit 1
source "./env.sh"
python3 jsonl2conditions.py "${args[@]}"
if [[ $? -eq 0 ]]; then
    filename="${jsonl_file##*/}"
    BATCH_NAME="${filename%.jsonl}"
    python3 conditions2cb0tokens.py --batch "$BATCH_NAME" && \
    python3 cb0tokens2tokens.py --batch "$BATCH_NAME" && \
    python3 tokens2audio.py --batch "$BATCH_NAME"
fi
