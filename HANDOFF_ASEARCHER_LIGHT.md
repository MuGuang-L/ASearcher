# ASearcher Light Handoff

This file records the changes added in this session so a new conversation can resume work quickly on another machine.

## Goal

We kept the original `ASearcher` code untouched and added a lightweight experimental path for:

- smaller-model multi-turn search-agent RL
- fewer search steps
- lower rollout/context cost
- better local observability without depending on W&B

The intended use case is a project-oriented, lightweight variant of `ASearcher` that is easier to run on limited compute and easier to inspect during training.

## New Files Added

### Lightweight training path

- [ASearcher/train/search_agent_light.py](/Users/longjianhao/mo/ASearcher/ASearcher/train/search_agent_light.py)
- [ASearcher/train/asearcher_light.py](/Users/longjianhao/mo/ASearcher/ASearcher/train/asearcher_light.py)

### Lightweight configs

- [ASearcher/configs/asearcher_local_light.yaml](/Users/longjianhao/mo/ASearcher/ASearcher/configs/asearcher_local_light.yaml)
- [ASearcher/configs/asearcher_local_light_flexible.yaml](/Users/longjianhao/mo/ASearcher/ASearcher/configs/asearcher_local_light_flexible.yaml)
- [ASearcher/configs/asearcher_local_light_qwen3.yaml](/Users/longjianhao/mo/ASearcher/ASearcher/configs/asearcher_local_light_qwen3.yaml)

### Qwen3 note

- [docs/qwen3_light_notes.md](/Users/longjianhao/mo/ASearcher/docs/qwen3_light_notes.md)

### Trace viewer

- [demo/light_trace_server.py](/Users/longjianhao/mo/ASearcher/demo/light_trace_server.py)
- [demo/light_trace_viewer.html](/Users/longjianhao/mo/ASearcher/demo/light_trace_viewer.html)
- [demo/light_trace_viewer.js](/Users/longjianhao/mo/ASearcher/demo/light_trace_viewer.js)
- [demo/light_trace_viewer.css](/Users/longjianhao/mo/ASearcher/demo/light_trace_viewer.css)

## What Was Changed

### 1. Lightweight search agent

`search_agent_light.py` is a new agent implementation derived from the original logic, but made cheaper to run.

Key reductions:

- search result snippets are shorter
- webpage content is heavily truncated
- webpage chunks are fewer and smaller
- rollout can use short summaries instead of full page text

Important parameters exposed:

- `max_doc_chars`
- `max_page_total_chars`
- `max_page_chunk_chars`
- `max_page_chunks`
- `short_page_preview_chars`
- `use_short_context_for_rollout`

### 2. Lightweight training entry

`asearcher_light.py` is a separate training entry that does not modify the original training script.

It adds:

- lighter defaults for search-agent RL
- optional step penalties for efficiency experiments
- repeated-action penalty
- richer rollout diagnostics

Important config fields supported by `AgentLightRLConfig`:

- `max_turns`
- `n_trajs`
- `topk`
- `max_doc_chars`
- `max_page_total_chars`
- `max_page_chunk_chars`
- `max_page_chunks`
- `use_short_context_for_rollout`
- `search_step_penalty`
- `access_step_penalty`
- `repeated_action_penalty`

### 3. Rich trace dumps for observability

The lightweight trainer now writes richer trace files in addition to the original generated rollout dumps.

Existing output:

- `generated/<version>/<qid>.jsonl`

New output:

- `generated/<version>/<qid>.trace.json`

The new trace file includes:

- episode-level status
- discard reason
- raw and normalized scores
- per-trajectory final score
- penalty breakdown
- final answer
- termination reason
- step-by-step tool usage
- LLM completion text previews

Special statuses:

- `status="skipped", reason="existing_dump_found"`
- `status="discarded", reason="all_normalized_scores_zero"`

### 4. Local trace viewer

The trace viewer is a small FastAPI + static HTML app for reading `*.trace.json` files.

It shows:

- all episodes found in a trace directory
- per-episode status and scores
- per-trajectory diagnostics
- step-by-step tool calls and tool-result previews
- final answers and penalty breakdowns

This is intended to help inspect:

- what the model did first
- why a rollout was kept or discarded
- how search/access penalties changed the reward
- where cost or bad behavior came from

## Config Summary

### `asearcher_local_light.yaml`

Fixed lightweight baseline config using:

- `Qwen/Qwen2.5-1.5B`

### `asearcher_local_light_flexible.yaml`

Flexible lightweight config.

You should replace:

- `actor.path: YOUR_MODEL_PATH_HERE`

Because `tokenizer_path`, `ref.path`, and `sglang.model_path` all reference `${actor.path}`, you only need to override one field.

### `asearcher_local_light_qwen3.yaml`

Lightweight config intended for:

- `Qwen/Qwen3-1.7B`

This uses smaller generation length than the generic lightweight config because Qwen3-style behavior may still be somewhat more verbose in practice.

## Qwen3 Note

The important point from [docs/qwen3_light_notes.md](/Users/longjianhao/mo/ASearcher/docs/qwen3_light_notes.md):

- the current lightweight training path does **not** use `tokenizer.apply_chat_template(..., enable_thinking=...)`
- it directly tokenizes prompt strings

So:

- using `Qwen3` here does not automatically mean the official thinking mode is enabled
- official Qwen3 `enable_thinking=False` would require a separate prompt-rendering path if we want strict support for it

## Suggested Models

Recommended current order for this lightweight project:

1. `Qwen/Qwen3-1.7B`
2. `Qwen/Qwen2.5-3B-Instruct`
3. `HuggingFaceTB/SmolLM3-3B`

Pragmatic recommendation:

- use `Qwen3-1.7B` as the main model for project packaging
- use `Qwen2.5-3B-Instruct` as a baseline

## Example Commands

### Lightweight training with flexible config

```bash
python3 -m areal.launcher.local ASearcher/train/asearcher_light.py \
  --config ASearcher/configs/asearcher_local_light_flexible.yaml \
  actor.path=Qwen/Qwen3-1.7B \
  train_dataset.path=/path/to/training_data.jsonl \
  experiment_name=asearcher-light-qwen3 \
  trial_name=run1
```

### Lightweight training with Qwen3 config

```bash
python3 -m areal.launcher.local ASearcher/train/asearcher_light.py \
  --config ASearcher/configs/asearcher_local_light_qwen3.yaml \
  train_dataset.path=/path/to/training_data.jsonl \
  experiment_name=asearcher-light-qwen3 \
  trial_name=run1
```

### Launch local trace viewer

```bash
python3 demo/light_trace_server.py \
  --trace-dir /path/to/generated \
  --host 127.0.0.1 \
  --port 8765
```

Then open:

```txt
http://127.0.0.1:8765
```

## Validation Already Done

The following Python files were syntax-checked successfully with `py_compile`:

- `ASearcher/train/asearcher_light.py`
- `demo/light_trace_server.py`

## Important Constraints

- Original files were not intentionally edited except for the new lightweight training entry itself.
- New work was added as separate files wherever possible.
- The user later mentioned the repository will be run on a server, so future additions should stay server-friendly:
  - no GUI dependency
  - no macOS-specific assumptions
  - command-line startup preferred
  - paths and ports configurable

## Good Next Steps

If work continues in a new conversation, the most useful follow-ups are:

1. Tune the lightweight config for the actual server GPU setup.
2. Add a stronger efficiency-focused reward preset.
3. Add experiment-comparison support to the trace viewer.
4. Add prompt/context-growth diagnostics to the trace dumps.
5. If needed, add a true Qwen3 non-thinking prompt-rendering path.

## One-Line Resume Prompt

If starting a new conversation, paste something like:

```txt
Continue from /Users/longjianhao/mo/ASearcher/HANDOFF_ASEARCHER_LIGHT.md and help me run/tune the lightweight ASearcher path on my server.
```
