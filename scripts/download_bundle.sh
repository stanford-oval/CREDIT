#!/bin/bash
# Download and unpack the released data into $CREDIT_ROOT: model outputs, evaluation items, and summary scores, i.e.
# everything scripts/08_paper_numbers.sh needs. The archives are hosted on OSF (https://osf.io/zm5ae/) and checked
# against data/SHA256SUMS in this repository.
#   ARCHIVES       archives to fetch (default: the four needed by 08_paper_numbers.sh)
#   OSF_VIEW_ONLY  key of an OSF view-only link, only needed while the project is not public
. "$(dirname "$0")/common.sh"
OSF_NODE="${OSF_NODE:-zm5ae}"
ARCHIVES="${ARCHIVES:-credit_items.tar.gz credit_outputs.tar.gz credit_outputs_extra.tar.gz credit_summaries.tar.gz}"
cd "$CREDIT_ROOT"
python3 - "$OSF_NODE" "${OSF_VIEW_ONLY:-}" <<'PY' > .osf_files.txt
import json, sys, urllib.request
node, key = sys.argv[1], sys.argv[2]
url = f"https://api.osf.io/v2/nodes/{node}/files/osfstorage/" + (f"?view_only={key}" if key else "")
while url:
    page = json.load(urllib.request.urlopen(url))
    for f in page["data"]:
        print(f["attributes"]["name"], f["links"]["download"])
    url = page["links"].get("next")
PY
for name in $ARCHIVES; do
  link="$(awk -v n="$name" '$1 == n {print $2}' .osf_files.txt)"
  [ -n "$link" ] || { echo "$name is not on OSF" >&2; exit 1; }
  [ -s "$name" ] || { log "downloading $name"; curl -fL --retry 3 -o "$name" "$link"; }
  sum="$(grep "  $name\$" "$HERE/data/SHA256SUMS")" || { echo "no checksum for $name in data/SHA256SUMS" >&2; exit 1; }
  echo "$sum" | shasum -a 256 -c -
done
# Skip macOS metadata files (._*) and a leftover temporary file in the archives; silence GNU tar's warnings about the
# macOS headers.
TAR_OPTS=(--exclude='._*' --exclude='*.tmp'); tar --version 2>/dev/null | grep -q GNU && TAR_OPTS+=(--warning=no-unknown-keyword)
for name in $ARCHIVES; do tar "${TAR_OPTS[@]}" -xzf "$name"; done
log "unpacked into $CREDIT_ROOT; next: bash scripts/08_paper_numbers.sh"
