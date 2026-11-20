#!/usr/bin/env bash
# Publish the current state of this folder to GitHub as a SINGLE commit.
# The remote keeps no history: every run replaces it with one fresh commit of the current files.
# (Same flow as EQUIV-public/.)
#
# One-time setup:  git init -b main && git remote add origin https://github.com/marmorera02/kvfold.git
# usage: bash publish_latest.sh ["commit message"]
set -euo pipefail

cd "$(dirname "$0")"

MSG="${1:-kvfold: surprise-gated folding and KV-state reload for long documents}"

# author identity for this repository only (GitHub noreply address, not a private email)
git config user.name "Christoph Deuel"
git config user.email "100279969+marmorera02@users.noreply.github.com"

git checkout -q --orphan publish_tmp          # new root commit, keeps the working tree and index
git add -A
git commit -q -F - <<EOF
$MSG

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
EOF
git branch -M main                            # replaces the old main locally
git push --force-with-lease origin main       # replaces the remote history with this one commit

# drop the old commits from the local object store too
git reflog expire --expire=now --all
git gc --prune=now -q

echo "--- published; commits on main: $(git rev-list --count main)"
git log --oneline -1
