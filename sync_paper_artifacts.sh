#!/bin/bash
# Preview (and optionally stage) the paper artifacts under runs/ that the
# .gitignore re-includes. Run on miyabi BEFORE committing.
#
# Why a guard: a multi-GB blob committed by accident stays in git history even
# after you delete it, and cleaning it means rewriting history on a repo that
# is already shared. Cheap to check first.
#
#   bash sync_paper_artifacts.sh          # preview only
#   bash sync_paper_artifacts.sh --stage  # preview, then git add
set -eo pipefail

LIMIT_KB=${LIMIT_KB:-2048}          # flag anything above this

mapfile -t FILES < <(
    { git ls-files --others --exclude-standard -- runs/
      git ls-files --modified -- runs/; } | sort -u)

if [ ${#FILES[@]} -eq 0 ]; then
    echo "[sync] nothing new or modified under runs/ — already committed?"
    exit 0
fi

echo "[sync] ${#FILES[@]} file(s) new/modified under runs/:"
printf '%s\0' "${FILES[@]}" | du -h --files0-from=- 2>/dev/null | sort -h

echo
big=0
for f in "${FILES[@]}"; do
    [ -f "$f" ] || continue
    kb=$(du -k "$f" | cut -f1)
    if [ "$kb" -gt "$LIMIT_KB" ]; then
        echo "  !! ${kb}KB  $f"
        big=1
    fi
done
if [ "$big" = 1 ]; then
    echo
    echo "[sync] ABOVE ${LIMIT_KB}KB — these are meant to be small text tables."
    echo "       Check the .gitignore re-includes at the bottom of the file"
    echo "       before staging; do not commit blobs into shared history."
    exit 1
fi

total=$(printf '%s\0' "${FILES[@]}" | du -ch --files0-from=- 2>/dev/null | tail -1 | cut -f1)
echo "[sync] total ${total} — all under ${LIMIT_KB}KB each. OK."

if [ "$1" = "--stage" ]; then
    git add -- runs/
    echo
    echo "[sync] staged. Now:"
    echo "  git commit -m 'Paper artifacts: tables, reports, examples'"
    echo "  git push origin main"
else
    echo
    echo "[sync] preview only. Re-run with --stage to git add."
fi
