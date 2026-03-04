# TODO

## High Priority

- [ ] HuggingFace token support for gated/private models (pass `HF_TOKEN` env var)
- [ ] Concurrent model downloads (currently limited to one at a time)
- [ ] Auto-restart instances on crash (configurable)
- [ ] Persist instance configurations across container restarts

## Features

- [ ] Model deletion from the UI
- [ ] Instance resource monitoring (GPU utilization, memory per instance)
- [ ] Configurable vLLM arguments per instance (additional CLI flags)
- [ ] Model search from HuggingFace Hub in the UI
- [ ] Instance naming (custom names instead of instance-1, instance-2)
- [ ] API key / basic auth for the admin UI
- [ ] docker-compose.yml / podman-compose.yml
- [ ] HTTPS support for admin UI

## UI Improvements

- [ ] Dark/light theme toggle
- [ ] Log filtering and search
- [ ] Log download/export
- [ ] GPU utilization charts over time
- [ ] Mobile-responsive layout improvements
- [ ] Toast notifications instead of alert() dialogs
- [ ] Confirmation dialog before stopping instances

## Technical Debt

- [ ] Add unit tests for vllm_manager.py
- [ ] Add integration tests for API endpoints
- [ ] Health check endpoint for the admin container itself
- [ ] Structured JSON logging
- [ ] Rate limiting on download endpoint
