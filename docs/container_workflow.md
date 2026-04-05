# Container Workflow

This document records the intended container-based development workflow for
this repository.

## Goal

Use Docker as the stable runtime layer for:

- CUDA / driver-compatible execution
- AReaL runtime dependencies
- training and retrieval server runs
- in-container Codex-based development

## Start the Container

From the host machine:

```bash
cd /home/ubuntu/ASearcher
bash scripts/pull_areal_runtime.sh
bash scripts/run_areal_docker.sh
```

This mounts the repository into the container at:

```txt
/workspace/ASearcher
```

Proxy variables from the host are forwarded automatically when present:

- `HTTP_PROXY`
- `HTTPS_PROXY`
- `ALL_PROXY`
- `NO_PROXY`
- and their lowercase variants

The startup script also tries to recover proxy exports from the invoking
user's `~/.bashrc` when launched via `sudo`, because `sudo` may clear the
interactive shell environment.

The container is launched with host networking by default on Linux so that
host-local proxy endpoints such as `127.0.0.1:6990/6991` remain reachable
inside the container.

For Node/npm-based tools such as Codex CLI, the startup script also forwards:

- `NPM_CONFIG_PROXY`
- `NPM_CONFIG_HTTPS_PROXY`

The default runtime image is:

```txt
ghcr.mirrorify.net/inclusionai/areal-runtime:v1.0.2-sglang
```

## Inside the Container

Move into the mounted repository:

```bash
cd /workspace/ASearcher
```

Recommended first checks:

```bash
python3 --version
python3 -m py_compile ASearcher/train/asearcher_light.py
```

## Codex in the Container

The intended workflow is to install and use Codex inside the container.

That means:

- Docker provides the runtime environment
- Codex runs inside that environment
- repository changes still persist on the host because `/workspace/ASearcher`
  is a bind mount

In practice:

```bash
cd /workspace/ASearcher
# install / authenticate Codex in the container
# then continue development here
```

## AReaL Setup Inside the Container

If the runtime image does not already expose the package in the expected way,
install vendored AReaL in editable mode:

```bash
cd /workspace/ASearcher/AReaL
uv pip install -e . --no-deps
```

## Recommended Working Directory

For day-to-day development, use:

```txt
/workspace/ASearcher
```

For AReaL-specific package work, use:

```txt
/workspace/ASearcher/AReaL
```

## Notes

- host-side file edits and container-side file edits affect the same repository
- training should preferentially run inside the container, not on the host
- the light local training path remains the most practical first target
