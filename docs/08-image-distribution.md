# 08 — Image distribution

How the cluster's container image gets built once and onto every node identically.

## Why identical images matter

A multi-node vLLM instance is not "several servers that cooperate" — it is **one** process group
whose members exchange tensors directly. Three layers all assume their peers are the same build:

- **Ray** version-checks on join. A worker whose `ray` differs from the head's is refused, or
  worse, joins and then deserializes an object it does not understand.
- **NCCL** negotiates collectives across the RoCE fabric. Mismatched NCCL/torch builds can
  complete the handshake and then hang in the middle of an all-reduce, with no error — the
  symptom is an instance that loads weights and then simply stops.
- **vLLM** splits the model by tensor/pipeline rank. Two vLLM versions can lay out or shard a
  model differently, so rank 0 and rank 1 disagree about what the bytes mean.

None of these failures announces itself as "version skew". They look like hardware faults, which
is the worst possible thing to be debugging on a cluster that also *has* hardware faults.

This is why the `Containerfile` pins the vLLM base **by digest**
(`docker.io/vllm/vllm-openai@sha256:0dc46f…`) and every pip package to an **exact** version
(`ray[default]==2.54.0`, not `>=`). A range lets two boxes built days apart resolve different
Ray/transformers builds. But exact pins only make the build reproducible *in principle*: wheels
get rebuilt, the Go toolchain moves, a base tag gets re-pushed. The only real guarantee is that
every node runs **the same artifact** — so we build once, push to a registry, and have nodes pull
it by digest.

## The flow

```
  Containerfile (GitHub)
        │
        │  built ONCE
        ├──────────────► .github/workflows/image.yml   (on push to main / v* / tag v*)
        └──────────────► scripts/push-image.sh         (from the admin box, after bash build.sh)
                                 │
                                 ▼
                 ghcr.io/inchix/vllm_manager:<tag>  ──►  @sha256:<digest>
                                 │
                                 │  IMAGE_REF=ghcr.io/inchix/vllm_manager@sha256:…
                                 ▼
                  admin node .env  ──►  GET /api/cluster/image
                                 │
                                 ▼
                  join.sh on a new node: podman pull <ref>
                                          podman tag  <ref> vllm-manager:latest
```

The admin node is the single source of truth for *which* image the cluster runs. `join.sh` asks
it (`GET /api/cluster/image`, authenticated with the join token) rather than guessing, and pulls
that exact ref. If `IMAGE_REF` is empty, `join.sh` falls back to `bash build.sh` on the joining
node — much slower, and it reintroduces exactly the skew described above. Treat the fallback as a
bootstrap convenience, not a supported mode.

**The repo is public, so the GHCR package is public**: nodes need no registry credentials to
pull. Only *pushing* requires auth.

### Tags vs. digests

The pipeline pushes several tags — `latest` (default branch only), the branch name, the short
SHA, and the git tag when tagged. Those are for humans. **Pin `IMAGE_REF` to the digest.** A tag
can be re-pushed to point at different bytes tomorrow, at which point a node that restarts and
re-pulls silently diverges from its peers. A digest cannot.

## Cutting a new image version

Do this whenever the `Containerfile` changes — a new base digest, a bumped pin, new `admin/` code
baked into the image.

### Via CI (preferred, once it fits on the runner)

```bash
git tag v0.4.1 && git push origin v0.4.1     # or just push to main / a v* branch
```

The `image` workflow builds and pushes, then prints the digest ref in the **job summary**. Copy
it from there.

> **Disk space caveat.** GitHub-hosted runners start with ~14 GB free and this image is
> ~10-16 GB. The workflow frees ~25-30 GB first (removing the preinstalled .NET, Android, GHC and
> CodeQL toolchains) before building. If it still runs out, do not work around it — move the job
> to a self-hosted runner on the admin box; the workflow has the instructions in a comment.

### From the admin box (the path that always works)

The admin box already builds this image and has the disk for it.

```bash
bash build.sh                                          # produces vllm-manager:latest

# one-time: log in with a PAT that has write:packages
#   https://github.com/settings/tokens/new?scopes=write:packages
sudo podman login ghcr.io -u <your-github-username>

bash scripts/push-image.sh --sudo --tag v0.4.1
```

Notes:

- `--sudo` because `build.sh` defaults to `USE_SUDO=sudo`, so the image lives in **rootful**
  podman's store — and rootful/rootless podman keep separate credential stores, so the `login`
  above needs `sudo` too. The script detects this case and tells you.
- Without `--tag`, the version is derived from the current branch if it looks like `vX.Y.Z`.
- `--registry` defaults to `ghcr.io/inchix/vllm_manager`; `--dry-run` shows the plan without
  pushing anything.

Either way you end with a digest ref:

```
ghcr.io/inchix/vllm_manager@sha256:1a2b3c…
```

### Rolling it out

1. **Admin node** — set the new ref in `.env` and restart:

   ```bash
   # .env
   IMAGE_REF=ghcr.io/inchix/vllm_manager@sha256:1a2b3c…
   ```

   ```bash
   bash run.sh
   ```

2. **New nodes** — nothing to do. `join.sh` reads the new ref from the admin and pulls it.

3. **Existing nodes** — they do *not* auto-upgrade; a running node keeps its current image. Pull
   and restart each one:

   ```bash
   sudo podman pull ghcr.io/inchix/vllm_manager@sha256:1a2b3c…
   sudo podman tag  ghcr.io/inchix/vllm_manager@sha256:1a2b3c… vllm-manager:latest
   bash run.sh
   ```

   Roll the whole cluster in one pass. A half-upgraded cluster is precisely the skew case, and it
   will not announce itself — see the failure modes above.

4. **Verify** every node agrees:

   ```bash
   sudo podman image inspect --format '{{.Digest}}' vllm-manager:latest
   ```

   Same digest on every box, or stop and fix it before launching a distributed instance.

## Air-gapped fallback

A node with no route to ghcr.io can be fed the image directly from a node that has it. This
preserves the digest — it is the same artifact, moved by hand:

```bash
# on a node that already has the image
sudo podman save ghcr.io/inchix/vllm_manager@sha256:1a2b3c… \
  | ssh <node> 'sudo podman load'
```

For a slow or flaky link, stage it as a file instead (the stream is ~10-16 GB uncompressed):

```bash
sudo podman save -o /tmp/vllm-manager.tar ghcr.io/inchix/vllm_manager@sha256:1a2b3c…
rsync -P /tmp/vllm-manager.tar <node>:/tmp/
ssh <node> 'sudo podman load -i /tmp/vllm-manager.tar && rm -f /tmp/vllm-manager.tar'
```

Then tag it locally on the target so `run.sh` finds it:

```bash
sudo podman tag ghcr.io/inchix/vllm_manager@sha256:1a2b3c… vllm-manager:latest
```

Check the digest matches the source node before running anything distributed.
