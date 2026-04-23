# TODO

## High Priority

- [x] HuggingFace token support for gated/private models (pass `HF_TOKEN` env var)
- [ ] Concurrent model downloads (currently limited to one at a time)
- [x] Auto-restart instances on crash (configurable, exponential backoff)
- [x] Persist instance configurations across container restarts

## Features

- [x] Model deletion from the UI
- [ ] Instance resource monitoring (GPU utilization, memory per instance)
- [ ] Configurable vLLM arguments per instance (allowlisted flags — `extra_args` in `/api/start`)
- [ ] Model search from HuggingFace Hub in the UI
- [ ] Instance naming (custom names instead of instance-1, instance-2)
- [x] API key / basic auth for the admin UI (`AUTH_ENABLED`, `ADMIN_API_KEY`)
- [ ] docker-compose.yml / podman-compose.yml
- [ ] HTTPS support for admin UI (use a reverse proxy for now)

## UI Improvements

- [x] Dark/light theme toggle
- [x] Log filtering and search
- [x] Log download/export
- [x] GPU utilization charts over time
- [x] Mobile-responsive layout improvements
- [x] Toast notifications instead of alert() dialogs
- [x] Confirmation dialog before stopping instances
- [x] Login page + Sign out

## Security & Ops

- [x] API-key auth (header + cookie) with `AUTH_ENABLED=false` escape hatch for isolated on-prem
- [x] Bind admin port to loopback on host by default (`ADMIN_BIND_HOST`)
- [x] Safe `.env` parsing (no shell execution)
- [x] Allowlist for `extra_args` passed to vLLM
- [x] Download state race condition (asyncio.Lock)
- [x] Container hardening (`no-new-privileges`, `--cap-drop=ALL`, SELinux-ready)
- [x] Container HEALTHCHECK
- [x] systemd `Restart=on-failure`
- [ ] Rate limiting on login + download endpoints

## Technical Debt

- [ ] Add unit tests for vllm_manager.py
- [ ] Add integration tests for API endpoints
- [ ] Structured JSON logging
