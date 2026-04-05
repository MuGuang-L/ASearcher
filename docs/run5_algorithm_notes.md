# Run5 Algorithm Notes

This note captures the algorithm-side issues observed while inspecting the
`asearcher-light-qwen3/run5` training outputs on 2026-04-05.

## Run Status

- `run5` was not active during inspection.
- Existing outputs showed rollout traces up to version `109` and weight updates up to
  `weight_update_v110`.

## What Improved

- Episode mean score improved over training windows.
- Later versions had better average raw rollout scores than early versions.

## Main Failures Observed

### 1. Search-only behavior dominated

The model mostly learned:

1. issue one `<search>`
2. read snippets
3. answer immediately

It did **not** learn a stable `search -> access -> answer` pattern.

Observed during trace analysis:

- many positive trajectories had `search_calls > 0`
- essentially none had `access_calls > 0`

### 2. Too many discarded positive episodes

Many episodes were discarded because all 4 trajectories had identical raw scores,
which produced zero-centered normalized rewards.

Typical pattern:

- all 4 trajectories searched once
- all 4 gave the same answer
- all 4 were correct
- the episode was discarded as `all_normalized_scores_zero`

This means the run produced correct behavior that still generated no GRPO update
signal.

### 3. Access was not economically attractive

The then-current setup encouraged shortcut behavior:

- search snippets exposed substantial information
- reward depended only on final answer correctness
- `access` added step cost
- no evidence-chain bonus existed

The resulting local optimum was often:

- search once
- avoid access
- answer from snippets

### 4. Search snippets were often noisy

Some high-scoring examples appeared to succeed despite weak or off-topic snippet
previews. This suggests answer reward can sometimes hide retrieval quality issues.

Implication:

- high score does not necessarily mean good evidence use
- some “wins” may reflect brittle answer matching rather than grounded retrieval

## Root Causes

### Reward design

`SearchToolBox` scores the extracted `<answer>` string against the ground truth.
It does not require page access or evidence verification before granting reward.

### Prompt/context exposure

Search snippets were long enough that many single-hop and comparison questions
could be solved without opening a page.

### Relative-reward training

Because the trainer mean-centers rewards across 4 trajectories, identical success
across the group yields no training signal.

## Questions That Were Naturally Easy

A real subset of the dataset is snippet-solvable:

- same-country yes/no questions
- birth/death ordering
- simple “what year” lookups
- direct attribute comparisons

So the issue is not only reward shaping; task mix also contributes to the shortcut.

## Configuration Change Made After Inspection

To push the policy toward `search -> access -> answer`, the Qwen3 light config was
adjusted:

- `topk: 3 -> 2`
- `max_doc_chars: 1200 -> 320`
- `access_step_penalty: 0.02 -> 0.0`
- `no_access_penalty: 0.3` added

The light trainer was also updated so `no_access_penalty` and
`no_evidence_penalty` are configurable instead of hard-coded.

## Follow-up Checks For Next Run

For the next run, inspect these first:

- `num_pages`
- `num_success_url_accesses`
- fraction of discarded episodes
- fraction of positive-but-uniform discarded episodes
- whether positive trajectories now include real `access` steps

## Working Hypothesis

The current bottleneck is not “the model cannot search”.
The bottleneck is:

- snippet-only answers are too cheap
- access is under-rewarded
- GRPO loses signal when all trajectories behave the same way

That should be the baseline assumption for future tuning.
