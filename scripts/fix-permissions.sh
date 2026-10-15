#!/usr/bin/env bash
# Make the workspace readable/writable by all users (host and container alike).
# Run once after cloning/copying to a new machine, or whenever a foreign UID left files behind:
#   sudo scripts/fix-permissions.sh
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"

chmod -R a+rwX "$root"
# New files inherit rw for everyone even if a process forgets to set umask (needs ACL support).
if command -v setfacl >/dev/null 2>&1; then
  find "$root" -type d -exec setfacl -m d:u::rwx,d:g::rwx,d:o::rwx {} + 2>/dev/null || true
fi
# Keep git objects world-writable and trust the repo regardless of owner.
# Write by file path: git refuses `-C` on a repo owned by another UID (e.g. root running this script).
git config --file "$root/.git/config" core.sharedRepository 0666
git config --system --get-all safe.directory 2>/dev/null | grep -qxF "$root" \
  || git config --system --add safe.directory "$root"
echo "permissions fixed under $root"
