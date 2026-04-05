# Qwen3 Light Notes

This note explains how to think about Qwen3 in the lightweight ASearcher setup.

## 1. Why there is no `enable_thinking` flag in `asearcher_light.py`

The lightweight training entry at
[ASearcher/train/asearcher_light.py](/Users/longjianhao/mo/ASearcher/ASearcher/train/asearcher_light.py)
does **not** call `tokenizer.apply_chat_template(...)`.

Instead, it directly tokenizes the plain prompt string:

- prompt construction happens in
  [ASearcher/train/asearcher_light.py](/Users/longjianhao/mo/ASearcher/ASearcher/train/asearcher_light.py)
- tokenization happens via `tokenizer(prompt, add_special_tokens=False)`

Because of that, the official `Qwen3` chat-template switch
`enable_thinking=True/False` is **not directly used** in this training path.

## 2. What this means in practice

- Using `Qwen3` here does **not** automatically mean the official chat-template
  thinking mode is enabled.
- The model can still produce longer reasoning traces if your prompt style
  encourages them, but this is different from the official chat-template
  thinking toggle.
- For this repository's lightweight search-agent experiments, the dominant
  cost comes from:
  - multi-turn rollout length
  - search/access tool calls
  - webpage context growth
  - number of sampled trajectories

## 3. Why `Qwen3-1.7B` is still a good choice here

- `AReaL` already includes a search-agent example trained with
  `Qwen/Qwen3-1.7B`
- it gives you a newer model family for project packaging
- it is still small enough for a lightweight setup

## 4. Recommended usage

Use the dedicated config:

- [ASearcher/configs/asearcher_local_light_qwen3.yaml](/Users/longjianhao/mo/ASearcher/ASearcher/configs/asearcher_local_light_qwen3.yaml)

Or use the flexible config and override the model path:

```bash
python3 -m areal.launcher.local ASearcher/train/asearcher_light.py \
  --config ASearcher/configs/asearcher_local_light_flexible.yaml \
  actor.path=Qwen/Qwen3-1.7B \
  train_dataset.path=/path/to/training_data.jsonl \
  experiment_name=asearcher-light-qwen3 \
  trial_name=run1
```

## 5. If you later want true official Qwen3 non-thinking mode

You would need a separate path that renders prompts with Qwen3's official
chat template, for example through:

- `tokenizer.apply_chat_template(..., enable_thinking=False)`

That would be a different inference/prompting path from the current
lightweight ASearcher workflow.
