# --- Stage 1: build modelfsd (v0.4.0 storage role: read-only NFSv3/TCP model server) ---
# A small standalone Go binary; built here so the final image can run the `storage` role
# with no Go toolchain at runtime. See storage/modelfsd/ and docs/03-storage-modelfsd.md.
FROM docker.io/library/golang:1.24 AS modelfsd-build
WORKDIR /src
COPY storage/modelfsd/ /src/
RUN go build -trimpath -o /out/modelfsd ./cmd/modelfsd

# --- Stage 2: the admin/agent image ---
# Pinned by digest for reproducibility. Every node in a multi-node cluster MUST
# run the byte-identical image (matching vLLM/NCCL); a floating tag would drift
# between boxes.
#
# IMPORTANT — Volta (V100 / sm_70) constraint:
# This is the STABLE `vllm/vllm-openai:latest` (vLLM 0.17.1, torch 2.10+cu129).
# The current `:nightly` ships torch built against CUDA 13 (cu130), and CUDA 13
# DROPPED Volta — it fails on V100 with "no kernel image is available for
# execution on the device". cu129 still includes sm_70 kernels, so stay on a
# CUDA-12 build for as long as this cluster runs V100s. Newer vLLM needs
# Ampere+ (sm_80+). To move up on Volta-compatible hardware, pick a newer
# digest whose torch is still cu12x and whose arch_list includes sm_70.
#   vllm/vllm-openai:latest pinned 2026-07-29 (vLLM 0.17.1, torch 2.10+cu129)
FROM docker.io/vllm/vllm-openai@sha256:0dc46f74eb0e630675d83101dc66c6441c4475cceedcf9235ee42b87c3affd23

RUN pip install --no-cache-dir nvidia-ml-py \
    && pip install --no-cache-dir "transformers>=5.5,<6" \
    && pip install --no-cache-dir "ray[default]>=2.9" \
    && pip install --no-cache-dir "mistral_common>=1.11.5" \
    && pip install --no-cache-dir "websockets>=12"
# websockets: the v0.4.0 control-plane agent (admin/agent/agentd.py) dials the admin's CCP
# WebSocket, and uvicorn needs it to serve that endpoint. Usually present via uvicorn[standard];
# pinned here so the agent works regardless of the base image's extras.
# mistral_common>=1.11.5: transformers 5.14's tokenization_mistral_common gates
# its imports on is_mistral_common_available() which requires >=1.11.5; the base
# image ships 1.9.1, so Mistral/Devstral (tekken tokenizer) models fail to load
# with "NameError: SpecialTokens" without this bump.

COPY admin/ /app/admin/
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# modelfsd binary for the storage role (built in stage 1). On PATH so the agent's
# runner finds it as `modelfsd` (override with MODELFSD_BIN).
COPY --from=modelfsd-build /out/modelfsd /usr/local/bin/modelfsd

WORKDIR /app

EXPOSE 8001-8010 7080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7080/healthz', timeout=3).status == 200 else 1)" || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
