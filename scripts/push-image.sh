#!/usr/bin/env bash
# Push a locally-built vLLM Manager image to the cluster's registry.
#
# WHY: every node in the cluster must run a BYTE-IDENTICAL image — Ray/NCCL/vLLM
# version skew between boxes breaks distributed inference in obscure ways. So the
# image is built ONCE and every node pulls that exact artifact. This script is the
# "build on the admin box" path; CI (.github/workflows/image.yml) does the same job
# on push, but GitHub-hosted runners are tight on disk for a ~10-16 GB image.
#
# After pushing it prints the DIGEST ref. That digest — not a tag — is what you put
# in the admin's .env as IMAGE_REF, because tags move and digests do not.
#
# Usage:
#   bash scripts/push-image.sh [--tag vX.Y.Z] [--registry REPO] [--runtime podman|docker]
#
# Options:
#   --tag vX.Y.Z      Version tag to push alongside :latest.
#                     Default: the current git branch, if it looks like vX.Y.Z.
#   --registry REPO   Target repository. Default: ghcr.io/inchix/vllm_manager
#   --runtime NAME    podman | docker. Default: podman if present, else docker.
#   --local NAME:TAG  Local image to push. Default: $IMAGE_NAME or vllm-manager:latest
#   --sudo            Run the container runtime under sudo (rootful podman — this is
#                     where build.sh puts the image when USE_SUDO=sudo, the default
#                     in .env.example). Also honours USE_SUDO=sudo from the env.
#   --skip-login-check  Push without first confirming a registry login.
#   -n, --dry-run     Print what would be tagged/pushed and exit.
#   -h, --help        This help.
#
# See docs/08-image-distribution.md.

set -euo pipefail

DEFAULT_REGISTRY="ghcr.io/inchix/vllm_manager"

REGISTRY=""
TAG=""
RUNTIME="${CONTAINER_RUNTIME:-}"
LOCAL_IMAGE="${IMAGE_NAME:-vllm-manager:latest}"
USE_SUDO="${USE_SUDO:-}"
SKIP_LOGIN_CHECK=0
DRY_RUN=0

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
say() { printf '\n== %s\n' "$*"; }
run() {
  printf '   $ %s\n' "$*"
  [ "$DRY_RUN" -eq 1 ] && return 0
  "$@"
}

usage() { sed -n '2,30p' "$0"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --tag)              TAG="${2:-}"; shift 2 ;;
    --registry)         REGISTRY="${2:-}"; shift 2 ;;
    --runtime)          RUNTIME="${2:-}"; shift 2 ;;
    --local)            LOCAL_IMAGE="${2:-}"; shift 2 ;;
    --sudo)             USE_SUDO="sudo"; shift ;;
    --skip-login-check) SKIP_LOGIN_CHECK=1; shift ;;
    -n|--dry-run)       DRY_RUN=1; shift ;;
    -h|--help)          usage; exit 0 ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
done

# --- resolve the runtime -----------------------------------------------------
if [ -z "$RUNTIME" ]; then
  for r in podman docker; do
    command -v "$r" >/dev/null 2>&1 && { RUNTIME="$r"; break; }
  done
fi
[ -n "$RUNTIME" ] || die "no container runtime found; install podman or docker, or pass --runtime"
command -v "$RUNTIME" >/dev/null 2>&1 || die "container runtime not found on PATH: $RUNTIME"
case "$RUNTIME" in
  podman|docker) : ;;
  *) die "--runtime must be podman or docker (got: $RUNTIME)" ;;
esac

# shellcheck disable=SC2206  # deliberate word split: optional sudo prefix
CMD=(${USE_SUDO:+"$USE_SUDO"} "$RUNTIME")

# --- resolve the registry ----------------------------------------------------
# Never push to a guessed target: resolve it explicitly and print it before acting.
REGISTRY="${REGISTRY:-${IMAGE_REGISTRY:-$DEFAULT_REGISTRY}}"
case "$REGISTRY" in
  */*) : ;;
  *) die "--registry must be a full repository like ghcr.io/inchix/vllm_manager (got: $REGISTRY)" ;;
esac
# Registry paths must be lowercase or the push is rejected.
[ "$REGISTRY" = "$(printf '%s' "$REGISTRY" | tr '[:upper:]' '[:lower:]')" ] \
  || die "registry repository must be lowercase: $REGISTRY"
REGISTRY_HOST="${REGISTRY%%/*}"

# --- resolve the version tag -------------------------------------------------
if [ -z "$TAG" ]; then
  branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  case "$branch" in
    v[0-9]*) TAG="$branch" ;;
    *)
      die "could not derive a version tag from the current branch ('${branch:-unknown}').
       Pass one explicitly, e.g.:  bash scripts/push-image.sh --tag v0.4.0"
      ;;
  esac
fi
case "$TAG" in
  latest) die "--tag latest is redundant: :latest is always pushed alongside the version tag" ;;
  *[!A-Za-z0-9._-]*) die "invalid tag (letters, digits, . _ - only): $TAG" ;;
esac

REMOTE_LATEST="$REGISTRY:latest"
REMOTE_TAGGED="$REGISTRY:$TAG"

say "Plan"
cat <<EOF
   runtime:   ${CMD[*]}
   local:     $LOCAL_IMAGE
   push:      $REMOTE_TAGGED
              $REMOTE_LATEST
EOF
[ "$DRY_RUN" -eq 1 ] && echo "   (dry run — nothing will be tagged or pushed)" || true

# --- verify the local image exists ------------------------------------------
say "Checking the local image"
image_exists() { "$@" image inspect "$LOCAL_IMAGE" >/dev/null 2>&1; }

if image_exists "${CMD[@]}"; then
  echo "   found: $LOCAL_IMAGE"
elif [ -z "$USE_SUDO" ] && sudo -n true >/dev/null 2>&1 && image_exists sudo "$RUNTIME"; then
  die "'$LOCAL_IMAGE' is not in your user's image store, but it IS in root's.
       build.sh uses USE_SUDO=sudo by default, so the image belongs to rootful $RUNTIME.
       Re-run this script the same way:
           sudo -v && bash scripts/push-image.sh --sudo --tag $TAG"
else
  die "local image not found: $LOCAL_IMAGE
       Build it first (on this box):
           bash build.sh
       …or point at a different local image with --local NAME:TAG."
fi

# --- verify the registry login ----------------------------------------------
# GHCR pulls from a PUBLIC repo need no auth, but PUSHING always does.
login_hint() {
  cat >&2 <<EOF

       You are not logged in to $REGISTRY_HOST (or the login could not be confirmed).

       1. Create a GitHub personal access token (classic) with the
          'write:packages' scope:
              https://github.com/settings/tokens/new?scopes=write:packages
       2. Log in (paste the token at the password prompt):
              ${CMD[*]} login $REGISTRY_HOST -u <your-github-username>

       Note: with --sudo, the login must ALSO be done under sudo — rootful and
       rootless $RUNTIME keep separate credential stores.

       Re-run with --skip-login-check to push anyway.
EOF
}

confirm_login() {
  # podman can answer directly.
  if [ "$RUNTIME" = "podman" ]; then
    "${CMD[@]}" login --get-login "$REGISTRY_HOST" >/dev/null 2>&1 && return 0
    return 1
  fi
  # docker has no --get-login; look for the host in its credential config.
  local cfg="${DOCKER_CONFIG:-$HOME/.docker}/config.json"
  [ -f "$cfg" ] && grep -qF "$REGISTRY_HOST" "$cfg" && return 0
  return 1
}

say "Checking the registry login"
if [ "$SKIP_LOGIN_CHECK" -eq 1 ]; then
  echo "   skipped (--skip-login-check)"
elif confirm_login; then
  echo "   logged in to $REGISTRY_HOST"
else
  login_hint
  die "not logged in to $REGISTRY_HOST"
fi

# --- tag and push ------------------------------------------------------------
say "Tagging"
run "${CMD[@]}" tag "$LOCAL_IMAGE" "$REMOTE_TAGGED"
run "${CMD[@]}" tag "$LOCAL_IMAGE" "$REMOTE_LATEST"

DIGEST_FILE=""
cleanup() { [ -n "$DIGEST_FILE" ] && rm -f "$DIGEST_FILE" || true; }
trap cleanup EXIT

say "Pushing $REMOTE_TAGGED"
if [ "$RUNTIME" = "podman" ] && [ "$DRY_RUN" -eq 0 ]; then
  # --digestfile is authoritative: it is the digest the registry actually stored.
  DIGEST_FILE="$(mktemp)"
  run "${CMD[@]}" push --digestfile "$DIGEST_FILE" "$REMOTE_TAGGED"
else
  run "${CMD[@]}" push "$REMOTE_TAGGED"
fi

say "Pushing $REMOTE_LATEST"
run "${CMD[@]}" push "$REMOTE_LATEST"

if [ "$DRY_RUN" -eq 1 ]; then
  say "Dry run complete — nothing was pushed."
  exit 0
fi

# --- resolve the digest ------------------------------------------------------
# The digest is the point of the exercise: tags move, digests do not. Nodes pin to
# the digest so the whole cluster provably runs the same bytes.
say "Resolving the pushed digest"
DIGEST=""
if [ -n "$DIGEST_FILE" ] && [ -s "$DIGEST_FILE" ]; then
  DIGEST="$(tr -d '[:space:]' < "$DIGEST_FILE")"
fi
if [ -z "$DIGEST" ]; then
  # Fall back to the repo digest the runtime recorded for this image.
  DIGEST="$("${CMD[@]}" image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' \
              "$LOCAL_IMAGE" 2>/dev/null \
            | sed -n "s|^${REGISTRY}@||p" | head -n1 || true)"
fi
if [ -z "$DIGEST" ]; then
  # Last resort: the manifest digest the runtime records for the pushed tag.
  # (podman exposes .Digest; docker does not, in which case we fall through to
  # the warning below rather than printing a digest we are not sure of.)
  DIGEST="$("${CMD[@]}" image inspect --format '{{.Digest}}' "$REMOTE_TAGGED" 2>/dev/null \
            | grep -Eo '^sha256:[0-9a-f]{64}$' | head -n1 || true)"
fi

if [ -z "$DIGEST" ]; then
  cat <<EOF

WARNING: pushed successfully, but could not resolve the digest automatically.
Find it with:
    ${CMD[*]} image inspect --format '{{.Digest}}' $REMOTE_TAGGED
…and then set IMAGE_REF to "$REGISTRY@<that digest>".
EOF
  exit 0
fi

DIGEST_REF="$REGISTRY@$DIGEST"

cat <<EOF

== Done

Pushed:
    $REMOTE_TAGGED
    $REMOTE_LATEST

Digest ref (this is the one that matters):
    $DIGEST_REF

NEXT: pin the cluster to it. On the ADMIN node, set in .env:

    IMAGE_REF=$DIGEST_REF

and restart the admin (bash run.sh). The admin advertises that ref at
GET /api/cluster/image, and join.sh pulls exactly it on every joining node —
so the whole cluster provably runs the same bytes. Existing nodes need to
re-pull and restart to pick up a new image; see docs/08-image-distribution.md.
EOF
