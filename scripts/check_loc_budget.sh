#!/usr/bin/env bash
# Scope gate (26Q3-REPO guardrail 2): non-test Python source must stay within
# LOC_BUDGET. Raising the budget means editing LOC_BUDGET in the same diff:
# visible and reviewable, never silent.
#
# Mirrors flaime-serving's gate (REPO-01), minus the vendored/ exemption —
# the demo repo vendors nothing, it consumes flaime-serving as a package. That
# also means there is no tamper gate here; the pinned rev plays that role.
set -euo pipefail
cd "$(dirname "$0")/.."
budget=$(tr -d '[:space:]' < LOC_BUDGET)
actual=$(find flaime_demo -name '*.py' -print0 | xargs -0 --no-run-if-empty cat | wc -l)
if [ "$actual" -gt "$budget" ]; then
  echo "LoC budget exceeded: $actual / $budget non-test source lines (see LOC_BUDGET)"
  exit 1
fi
echo "LoC budget OK: $actual / $budget"
