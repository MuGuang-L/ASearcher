# Repository Direction

This repository is no longer treated as a pristine upstream checkout of
`ASearcher`.

The working direction is:

- keep the original search-agent training logic available
- vendor `AReaL` inside this repository for local iteration
- prioritize the lightweight local path as the practical default
- turn ad hoc server assumptions into repository-owned scripts and configs
- reduce dependency on old upstream path conventions and stale API assumptions

## Current Decisions

### 1. Vendored runtime dependency

`AReaL` is checked into:

```txt
/home/ubuntu/ASearcher/AReaL
```

This keeps the training stack inspectable and patchable from one workspace.

### 2. Lightweight path is the main practical path

For constrained hardware and faster iteration, the most useful entry today is:

```txt
ASearcher/train/asearcher_light.py
```

paired with:

```txt
ASearcher/configs/asearcher_local_light_qwen3.yaml
```

### 3. Repository-owned launch surface

The intended human-facing entrypoints are:

- `scripts/build_index.sh`
- `scripts/launch_local_server.sh`
- `scripts/run_light_local.sh`
- `scripts/run_areal_docker.sh`

These should continue to absorb machine-specific setup details so the training
code stays focused on RL logic.

## Known Transitional State

The repository is still in a transition phase. It is not yet a cleanly renamed
or fully restructured RL platform.

Examples of remaining transition artifacts:

- top-level README still contains upstream ASearcher benchmark/release framing
- training code still uses upstream file/module names
- some configs still preserve upstream naming even though the local layout has changed
- `AReaL` is currently vendored as a directory, not yet managed as a submodule or pinned fork

## Recommended Next Steps

### Near term

- validate one full lightweight local training run end to end
- validate one full local retrieval server run end to end
- decide whether vendored `AReaL` should remain a plain checkout or become a pinned submodule
- add one smoke-test script that checks imports, config loading, and launcher entrypoints

### Medium term

- create a top-level project name and replace `ASearcher`-specific branding in README/docs
- split "upstream preserved code" from "local platform extensions"
- consolidate duplicated training-loop logic across `asearcher.py`,
  `asearcher_light.py`, and `asearcher_reasoning.py`
- introduce a small compatibility layer for AReaL APIs instead of patching call sites repeatedly

### Long term

- define a stable internal interface for:
  - rollout workflow
  - reward shaping
  - tool backends
  - tracing and observability
  - local vs web search environments
- make this repository model-agnostic enough to host multiple RL agent projects,
  not just the original ASearcher family
