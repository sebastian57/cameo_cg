#! /bin/bash

watch -n 2 '
echo "=== develbooster ==="
sinfo -p develbooster
echo
echo "=== booster ==="
sinfo -p booster
echo "=== jobs ==="
squeue --me

'


