# Docker / Runtime Handoff

This file is the current handoff for continuing work inside the Docker
container with a new Codex session.

## Current Goal

Turn this repository into a maintainable, self-owned RL codebase based on:

- the original ASearcher training logic
- a vendored `AReaL` checkout
- a container-first workflow
- the lightweight local training path as the main practical starting point

## What Has Already Been Done

### 1. Vendored AReaL checkout

`AReaL` has been cloned into:

```txt
/home/ubuntu/ASearcher/AReaL
```

### 2. ASearcher compatibility fixes against newer AReaL

Confirmed compatibility edits were made in:

- `/home/ubuntu/ASearcher/ASearcher/train/asearcher.py`
- `/home/ubuntu/ASearcher/ASearcher/train/asearcher_light.py`
- `/home/ubuntu/ASearcher/ASearcher/train/asearcher_reasoning.py`
- `/home/ubuntu/ASearcher/ASearcher/utils/search_tool.py`

These changes include:

- replacing removed import paths
- aligning weight-update flow with newer AReaL usage
- fixing `StatsLogger` construction and logging calls
- removing stale `realhf` and old redistributor assumptions

### 3. Configs were localized to this machine/workspace

Several configs now use repository-local assumptions such as:

```txt
PYTHONPATH=/home/ubuntu/ASearcher:/home/ubuntu/ASearcher/AReaL
RAG_SERVER_ADDR_DIR=/tmp/areal/rag_server_addrs
```

### 4. Repository scripts were added / improved

Relevant scripts now present:

- `/home/ubuntu/ASearcher/scripts/install_docker_ubuntu.sh`
- `/home/ubuntu/ASearcher/scripts/configure_docker_mirrors.sh`
- `/home/ubuntu/ASearcher/scripts/pull_areal_runtime.sh`
- `/home/ubuntu/ASearcher/scripts/run_areal_docker.sh`
- `/home/ubuntu/ASearcher/scripts/run_light_local.sh`
- `/home/ubuntu/ASearcher/scripts/build_index.sh`
- `/home/ubuntu/ASearcher/scripts/launch_local_server.sh`

### 5. Docker pull path was fixed

The original `ghcr.io` path was too unreliable on this machine.

The runtime image was successfully pulled from:

```txt
ghcr.mirrorify.net/inclusionai/areal-runtime:v1.0.2-sglang
```

The image was also tagged locally as:

```txt
ghcr.io/inclusionai/areal-runtime:v1.0.2-sglang
```

Current observed image size:

- about `99.6GB`

### 6. NVIDIA container runtime was installed and configured

The host originally failed with:

```txt
failed to discover GPU vendor from CDI: no known GPU vendor found
```

This was fixed by:

- installing `nvidia-container-toolkit`
- running `nvidia-ctk runtime configure --runtime=docker`
- restarting Docker

Validation already done:

- Docker runtime list includes `nvidia`
- CUDA test container ran `nvidia-smi`
- PyTorch inside the AReaL runtime container reports:

```txt
torch.cuda.is_available() == True
device_count == 2
```

## Current Container State

The user successfully entered the container with:

```bash
cd /home/ubuntu/ASearcher
sudo bash scripts/run_areal_docker.sh
```

Inside container:

```txt
/workspace/ASearcher
```

Validated in-container commands:

```bash
pwd
python3 --version
python3 -c "import areal; print('areal ok')"
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Observed output:

- `pwd` => `/workspace/ASearcher`
- Python `3.12.3`
- `areal ok`
- `True 2`

## Additional Docker / Proxy Findings

The host machine uses local SSH-tunneled proxy endpoints:

```txt
http://127.0.0.1:6991
socks5h://127.0.0.1:6990
```

This matters because Docker bridge networking cannot reach the host loopback
address from inside the container.

To fix this, `scripts/run_areal_docker.sh` was updated to:

- use `--network host` by default
- forward proxy variables in both uppercase and lowercase forms
- forward npm proxy environment values as well
- recover proxy exports from the invoking user's `~/.bashrc` when launched with `sudo`

### Codex login status

The container can now reach OpenAI endpoints through the proxy.

Validated:

```bash
curl -I https://auth.openai.com/oauth/token
```

and:

```bash
curl -I https://api.openai.com
```

However, Codex CLI login was still reported by the user as failing during token
exchange.

This means infrastructure/network connectivity is much closer to correct, but
the login flow may still require one of:

- retrying after the proxy/environment fixes
- using `codex login --device-auth`
- using `codex login --with-api-key`

The new Codex session inside the container should verify this first.

## Important Observation About `import areal`

Inside the container, `import areal` already works before doing any editable
install from the mounted repo.

That means the runtime image already contains a preinstalled `areal` package.

This is important because:

- `/workspace/ASearcher/AReaL` is the mounted source checkout
- but Python may still be importing the image-bundled package instead of the
  mounted repo source

For continued development, this ambiguity should be resolved explicitly.

## Highest-Priority Next Steps For New Codex

### 1. Determine which `areal` is being imported

Inside the container, inspect:

```bash
python3 -c "import areal; print(areal.__file__)"
python3 -c "import sys; print('\\n'.join(sys.path))"
```

Goal:

- verify whether Python is using the image-bundled install
- or the mounted checkout

### 2. Prefer switching to editable install from mounted source

If the imported `areal` is not from `/workspace/ASearcher/AReaL`, switch to
editable install:

```bash
cd /workspace/ASearcher/AReaL
uv pip install -e . --no-deps
```

Then re-check:

```bash
python3 -c "import areal; print(areal.__file__)"
```

### 3. Re-validate the mounted source path

After editable install, verify:

```bash
python3 -c "import areal; print('areal ok')"
python3 -m py_compile /workspace/ASearcher/ASearcher/train/asearcher_light.py
```

### 4. Begin actual lightweight-path bring-up

The next real objective is no longer Docker setup. It is:

- getting the lightweight local training path closer to a real run

Main files:

- `/workspace/ASearcher/ASearcher/train/asearcher_light.py`
- `/workspace/ASearcher/ASearcher/train/search_agent_light.py`
- `/workspace/ASearcher/ASearcher/configs/asearcher_local_light_qwen3.yaml`

### 5. Validate end-to-end local workflow assumptions

Check these paths and assumptions next:

- local RAG server startup
- index-building assumptions
- config loading under container environment
- whether AReaL launcher behavior still matches these training scripts

## Suggested Immediate Command Sequence For New Codex

Inside container:

```bash
cd /workspace/ASearcher
python3 -c "import areal; print(areal.__file__)"
python3 -c "import sys; print('\\n'.join(sys.path))"
cd /workspace/ASearcher/AReaL
uv pip install -e . --no-deps
cd /workspace/ASearcher
python3 -c "import areal; print(areal.__file__)"
python3 -m py_compile /workspace/ASearcher/ASearcher/train/asearcher_light.py
```

Then continue with:

- validating the lightweight config path
- validating the local retrieval workflow
- attempting the first real lightweight training launch

## Related Documents

- `/home/ubuntu/ASearcher/HANDOFF_ASEARCHER_LIGHT.md`
- `/home/ubuntu/ASearcher/docs/repo_direction.md`
- `/home/ubuntu/ASearcher/docs/container_workflow.md`

## Summary

Docker, GPU runtime, and AReaL container bootstrapping are now solved.

The handoff point is no longer infrastructure rescue.

The next Codex should focus on:

- resolving `areal` source-of-truth inside the container
- locking the mounted `AReaL` checkout into the Python environment
- moving from environment bring-up to real lightweight ASearcher execution
