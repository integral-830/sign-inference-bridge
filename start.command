#!/bin/bash
# Double-click this file in Finder (or run ./start.command) to start everything.
# First time only: in Terminal run   chmod +x start.command   (the zip may remove the executable flag).
# Extra options are passed on:   ./start.command --no-window --token mysecret
cd "$(dirname "$0")" || exit 1
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
else
  echo "No .venv found here. Do the install step first (RUNBOOK.md Part 1)."
  exit 1
fi
python run.py "$@"
status=$?
echo
echo "(finished with status $status; press Enter to close)"
read -r _
