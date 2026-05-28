#!/bin/sh
set -eu

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

base_ref="${DOCPACT_BASE_REF:-origin/main}"
head_ref="${DOCPACT_HEAD_REF:-HEAD}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --base)
      base_ref="$2"
      shift 2
      ;;
    --head)
      head_ref="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

if ! base_sha="$(git merge-base "$head_ref" "$base_ref" 2>/dev/null)"; then
  echo "Could not resolve merge-base for $head_ref and $base_ref." >&2
  echo "Fetch the base ref or rerun with DOCPACT_BASE_REF=<ref>." >&2
  exit 2
fi

head_sha="$(git rev-parse "$head_ref")"

echo "Running ai-doc-lint unit tests."
node --test .github/scripts/ai-doc-lint.test.mjs

echo "Running ai-doc-lint: base=$base_sha head=$head_sha."
node .github/scripts/ai-doc-lint.mjs --mode enforce --base "$base_sha" --head "$head_sha"
