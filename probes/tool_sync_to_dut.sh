#!/usr/bin/env bash
# Push a local branch to the DUT's clone via a git bundle.
#
# The DUT has no credentials for the forks and the forks are not pushed, so the
# bundle is the transport. Bundles carry real commits, so SHAs on the DUT match
# the SHAs quoted in the write-up.
#
# usage: tool_sync_to_dut.sh <local-repo-path> <branch> <remote-clone-name>
set -euo pipefail

REPO="$1"
BRANCH="$2"
NAME="$3"
DUT="${DUT:?set DUT=user@host}"

BUNDLE="/tmp/${NAME}.bundle"
git -C "$REPO" bundle create "$BUNDLE" "$BRANCH" >/dev/null 2>&1
scp -q -o BatchMode=yes "$BUNDLE" "$DUT:~/dlrmv3-xpu/"

ssh -o BatchMode=yes "$DUT" "
  set -e
  cd ~/dlrmv3-xpu/src-${NAME}
  git fetch -q ~/dlrmv3-xpu/${NAME}.bundle '${BRANCH}:refs/remotes/bundle/${BRANCH}'
  git checkout -q -B '${BRANCH}' 'refs/remotes/bundle/${BRANCH}'
  echo \"  src-${NAME} now at \$(git rev-parse --short HEAD)  \$(git log -1 --format=%s | cut -c1-60)\"
"
