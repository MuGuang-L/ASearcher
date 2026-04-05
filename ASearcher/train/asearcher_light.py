import asyncio
import gc
import hashlib
import json
import os
from pathlib import Path
import uuid
from dataclasses import dataclass, field
from typing import List

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
import numpy as np
from tensordict import TensorDict
import torch
import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerFast

from areal.api.alloc_mode import _AllocationMode
from areal.api.cli_args import GRPOConfig, GenerationHyperparameters, load_expr_config
from areal.api.io_struct import (
    FinetuneSpec,
    ModelRequest,
    StepInfo,
    WeightUpdateMeta,
)
from areal.api.workflow_api import RolloutWorkflow
from areal.engine.fsdp_engine import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal import current_platform
from areal.utils import logging, seeding, stats_tracker
from areal.utils.data import (
    broadcast_tensor_container,
    concat_padded_tensors,
    cycle_dataloader,
    tensor_container_to,
)
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
try:
    from areal.utils.device import log_gpu_stats
except ImportError:
    def log_gpu_stats(*args, **kwargs):
        return None

from ASearcher.train.prompts import (
    INVALID_PROMPT,
    SEARCH_ACCESS_PROMPT_TEMPLATE,
    SEARCH_ONLY_PROMPT_TEMPLATE,
    VALID_PROMPT,
)
from ASearcher.train.search_agent_light import SearchAgentLight
from ASearcher.utils.rewards import correct_format_fn
from ASearcher.utils.search_tool import SearchToolBox

worker_id = uuid.uuid4().hex[:4]
logger = logging.getLogger(f"ASearcherLight @ {worker_id}")


def hash_numbers(numbers):
    return hashlib.sha256(json.dumps(numbers, sort_keys=True).encode()).hexdigest()


class ASearcherLightWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        dataset_path: str,
        dump_dir: str | None = None,
        max_turns: int = 12,
        n_trajs: int = 4,
        search_client_type: str = "async-search-access",
        reward_type: str = "F1",
        topk: int = 3,
        valid_inst_ratio: float = 1.0,
        max_tokens: int = 8192,
        search_only: bool = False,
        max_doc_chars: int = 1200,
        max_page_total_chars: int = 30000,
        max_page_chunk_chars: int = 6000,
        max_page_chunks: int = 3,
        short_page_preview_chars: int = 160,
        use_short_context_for_rollout: bool = True,
        search_step_penalty: float = 0.0,
        access_step_penalty: float = 0.0,
        repeated_action_penalty: float = 0.0,
        no_evidence_penalty: float = 0.2,
        no_access_penalty: float = 0.1,
    ):
        self.gconfig = gconfig
        self.gconfig.n_samples = 1
        self.tokenizer = tokenizer
        self.dump_dir = dump_dir
        self.max_tokens = max_tokens
        self.search_only = search_only
        self.max_turns = max_turns
        self.n_trajs = n_trajs
        self.reward_type = reward_type
        self.topk = topk
        self.valid_inst_ratio = valid_inst_ratio
        self.search_client_type = search_client_type
        self.agent_kwargs = dict(
            max_doc_chars=max_doc_chars,
            max_page_total_chars=max_page_total_chars,
            max_page_chunk_chars=max_page_chunk_chars,
            max_page_chunks=max_page_chunks,
            short_page_preview_chars=short_page_preview_chars,
            use_short_context_for_rollout=use_short_context_for_rollout,
        )
        self.search_step_penalty = search_step_penalty
        self.access_step_penalty = access_step_penalty
        self.repeated_action_penalty = repeated_action_penalty
        self.no_evidence_penalty = no_evidence_penalty
        self.no_access_penalty = no_access_penalty
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

        self.toolbox = SearchToolBox(
            dataset_path=dataset_path,
            reward_type=self.reward_type,
            topk=self.topk,
            search_client_type=self.search_client_type,
        )

    def _dump_episode_trace(self, *, version: int | str, qid: str, payload: dict) -> None:
        if self.dump_dir is None:
            return
        trace_dir = Path(self.dump_dir) / str(version)
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / f"{qid}.trace.json"
        with open(trace_path, "w") as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, indent=2)

    async def collect_agent_trajectory(
        self, valid_inst, qid, prompt, prompt_token_ids, engine, traj_idx: int
    ):
        agent = SearchAgentLight(prompt, prompt_token_ids, **self.agent_kwargs)
        score = 0.0
        ground_truth = None
        traj_rid = uuid.uuid4().hex
        seen_actions = set()
        search_calls = 0
        access_calls = 0
        repeat_calls = 0
        step_events = []
        terminated_reason = "max_turns_or_finished"
        logger.info(
            "Rollout start qid=%s traj=%s valid_inst=%s max_turns=%s",
            qid,
            traj_idx,
            valid_inst,
            self.max_turns,
        )

        while agent.num_turns < self.max_turns and not agent.is_finished:
            input_ids, sampling_params = agent.prepare_llm_query(self.tokenizer)
            req = ModelRequest(
                rid=traj_rid,
                input_ids=input_ids,
                gconfig=self.gconfig.new(n_samples=1),
            )
            if "stop" in sampling_params:
                req.gconfig.stop = sampling_params["stop"]
            if len(input_ids) + self.gconfig.max_new_tokens >= self.max_tokens:
                terminated_reason = "max_token_budget"
                logger.info(
                    "Rollout stop qid=%s traj=%s turn=%s reason=max_token_budget input_tokens=%s",
                    qid,
                    traj_idx,
                    agent.num_turns,
                    len(input_ids),
                )
                break

            logger.info(
                "Rollout gen qid=%s traj=%s turn=%s input_tokens=%s stop=%s",
                qid,
                traj_idx,
                agent.num_turns,
                len(input_ids),
                sampling_params.get("stop", []),
            )
            resp = await engine.agenerate(req)
            completion_str = self.tokenizer.decode(resp.output_tokens)
            tool_calls = agent.consume_llm_response(resp, completion_str)
            event = dict(
                turn=agent.num_turns,
                input_tokens=len(input_ids),
                output_tokens=len(resp.output_tokens),
                stop=sampling_params.get("stop", []),
                completion_text=completion_str,
                tool_call=None,
                tool_result_type=None,
                tool_result_preview=None,
                extracted_score=None,
            )

            if tool_calls:
                tool_call = tool_calls[0]
                logger.info(
                    "Rollout tool qid=%s traj=%s turn=%s tool=%s output_tokens=%s",
                    qid,
                    traj_idx,
                    agent.num_turns,
                    tool_call,
                    len(resp.output_tokens),
                )
                normalized_call = tool_call.strip().lower()
                if normalized_call in seen_actions:
                    repeat_calls += 1
                else:
                    seen_actions.add(normalized_call)
                if tool_call.startswith("<search>"):
                    search_calls += 1
                elif tool_call.startswith("<access>"):
                    access_calls += 1

                res = (await self.toolbox.step((qid, [tool_call])))[0]
                agent.consume_tool_response(res, topk=self.topk)
                logger.info(
                    "Rollout tool-result qid=%s traj=%s turn=%s type=%s score=%s has_page=%s docs=%s",
                    qid,
                    traj_idx,
                    agent.num_turns,
                    res.get("type"),
                    res.get("score"),
                    bool(res.get("page")),
                    len(res.get("documents") or []),
                )
                event["tool_call"] = tool_call
                event["tool_result_type"] = res.get("type")
                if res.get("type") == "access":
                    event["tool_result_preview"] = (res.get("page") or "")[:300]
                else:
                    docs = res.get("documents") or []
                    event["tool_result_preview"] = "\n".join(docs[:2])[:300]
                if "score" in res:
                    score = res["score"]
                    event["extracted_score"] = score
                if "ground_truth" in res:
                    ground_truth = res["ground_truth"]
            else:
                logger.info(
                    "Rollout no-tool qid=%s traj=%s turn=%s output_tokens=%s tail=%s",
                    qid,
                    traj_idx,
                    agent.num_turns,
                    len(resp.output_tokens),
                    completion_str[-120:],
                )

            step_events.append(event)
            if resp.output_tokens[-1] in [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]:
                terminated_reason = "eos_or_pad"
                logger.info(
                    "Rollout stop qid=%s traj=%s turn=%s reason=eos_or_pad",
                    qid,
                    traj_idx,
                    agent.num_turns,
                )
                break

        llm_gen_records = agent.memory.filter_records("llm_gen")
        format_reward = float(
            all(correct_format_fn(i, record.text) for i, record in enumerate(llm_gen_records))
        )
        base_score = (score or 0.0) * format_reward
        penalty_breakdown = dict(
            search=search_calls * self.search_step_penalty,
            access=access_calls * self.access_step_penalty,
            repeated=repeat_calls * self.repeated_action_penalty,
            no_evidence=0.0,
            no_access=0.0,
        )
        if valid_inst and search_calls == 0 and access_calls == 0:
            penalty_breakdown["no_evidence"] = self.no_evidence_penalty
        if (
            valid_inst
            and self.search_client_type == "async-search-access"
            and search_calls > 0
            and access_calls == 0
        ):
            penalty_breakdown["no_access"] = self.no_access_penalty
        score = base_score
        score -= penalty_breakdown["search"]
        score -= penalty_breakdown["access"]
        score -= penalty_breakdown["repeated"]
        score -= penalty_breakdown["no_evidence"]
        score -= penalty_breakdown["no_access"]

        pred_answer = agent.get_answer()
        judge_q_invalid = False
        if pred_answer is not None:
            judge_q_invalid = any(
                token in pred_answer for token in ["question", "invalid", "appropriate", "valid"]
            )
        if valid_inst and judge_q_invalid:
            score = -0.5

        stats = agent.memory.logging_stats()
        stats.update(
            dict(
                score=score,
                judge_q_invalid=judge_q_invalid,
                format_reward=format_reward,
                repeated_actions=repeat_calls,
            )
        )
        diagnostics = dict(
            qid=qid,
            traj_idx=traj_idx,
            valid_inst=valid_inst,
            final_answer=pred_answer,
            base_score=base_score,
            final_score=score,
            format_reward=format_reward,
            penalty_breakdown=penalty_breakdown,
            search_calls=search_calls,
            access_calls=access_calls,
            repeat_calls=repeat_calls,
            terminated_reason=terminated_reason,
            step_events=step_events,
        )
        logger.info(
            "Rollout done qid=%s traj=%s turns=%s score=%s base_score=%s format_reward=%s reason=%s repeated=%s",
            qid,
            traj_idx,
            len(llm_gen_records),
            score,
            base_score,
            format_reward,
            terminated_reason,
            repeat_calls,
        )
        return ground_truth, score, agent.memory, stats, diagnostics

    async def arun_episode(self, engine, data):
        qid = None
        for key in ["query_id", "id", "qid"]:
            qid = data.get(key, None)
            if qid is not None:
                break
        qid = str(qid) or uuid.uuid4().hex

        if self.dump_dir is not None:
            import glob

            pattern = os.path.join(self.dump_dir, "*", f"{qid}.jsonl")
            if len(glob.glob(pattern)) > 0:
                logger.info(f"{qid} is already trained on")
                self._dump_episode_trace(
                    version="skipped",
                    qid=qid,
                    payload=dict(
                        qid=qid,
                        status="skipped",
                        reason="existing_dump_found",
                        question=data.get("question"),
                    ),
                )
                return None

        version = engine.get_version()
        logger.info("Episode start qid=%s version=%s question=%s", qid, version, data.get("question"))
        prompt_template = (
            SEARCH_ONLY_PROMPT_TEMPLATE if self.search_only else SEARCH_ACCESS_PROMPT_TEMPLATE
        )
        prompt = prompt_template.replace("{question}", data["question"])
        valid_inst = np.random.uniform(0, 1) <= self.valid_inst_ratio
        if valid_inst:
            prompt = prompt.replace(INVALID_PROMPT, VALID_PROMPT)
        prompt_token_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        trajs = await asyncio.gather(
            *[
                self.collect_agent_trajectory(
                    valid_inst, qid, prompt, prompt_token_ids, engine, traj_idx
                )
                for traj_idx in range(self.n_trajs)
            ]
        )

        ground_truth, scores, results, stats, diagnostics = None, [], [], [], []
        for gt, score, traj, traj_stats, traj_diag in trajs:
            if gt is not None:
                ground_truth = gt
            scores.append(score)
            stats.append(traj_stats)
            diagnostics.append(traj_diag)

        raw_scores = scores
        score_mean = np.asarray(scores).mean()
        scores = [score - score_mean for score in scores]
        trace_payload = dict(
            qid=qid,
            version=version,
            status="ok",
            question=data.get("question"),
            valid_inst=valid_inst,
            prompt=prompt,
            raw_scores=raw_scores,
            normalized_scores=scores,
            mean_score=score_mean,
            ground_truth=ground_truth,
            trajectories=[],
        )
        if all(score == 0 for score in scores):
            trace_payload["status"] = "discarded"
            trace_payload["reason"] = "all_normalized_scores_zero"
            trace_payload["trajectories"] = diagnostics
            logger.info(
                "Episode discarded qid=%s version=%s raw_scores=%s normalized_scores=%s reason=%s",
                qid,
                version,
                raw_scores,
                scores,
                trace_payload["reason"],
            )
            self._dump_episode_trace(version=version, qid=qid, payload=trace_payload)
            return None

        traj_memories = [traj for _, _, traj, _, _ in trajs]
        for i, traj_memory in enumerate(traj_memories):
            seqs = []
            for record in traj_memory.memory:
                if record.type != "llm_gen":
                    continue
                success = False
                for seq in seqs:
                    if record.input_len < len(seq["input_ids"]):
                        continue
                    h_cur = hash_numbers(record.input_tokens[: len(seq["input_ids"])])
                    h_seq = hash_numbers(seq["input_ids"])
                    if h_cur == h_seq:
                        seq_len = len(seq["input_ids"])
                        seq["input_ids"] = record.input_tokens + record.output_tokens
                        seq["logprobs"] += [0.0] * (record.input_len - seq_len) + record.output_logprobs
                        seq["loss_mask"] += [0] * (record.input_len - seq_len) + [1] * record.output_len
                        seq["versions"] += [-1] * (record.input_len - seq_len) + record.output_versions
                        success = True
                        break
                if not success:
                    seqs.append(
                        dict(
                            input_ids=record.input_tokens + record.output_tokens,
                            logprobs=[0.0] * record.input_len + record.output_logprobs,
                            loss_mask=[0] * record.input_len + [1] * record.output_len,
                            versions=[-1] * record.input_len + record.output_versions,
                        )
                    )

            traj_stats = stats.pop(0)
            traj_diag = diagnostics[i]
            traj_diag["normalized_score"] = scores[i]
            traj_diag["memory_records"] = len(traj_memory.memory)
            traj_diag["llm_turns"] = len([r for r in traj_memory.memory if r.type == "llm_gen"])
            first_llm_gen = True
            for seq in seqs:
                item = dict(
                    input_ids=torch.tensor(seq["input_ids"]).unsqueeze(0),
                    loss_mask=torch.tensor(seq["loss_mask"]).unsqueeze(0),
                    logprobs=torch.tensor(seq["logprobs"]).unsqueeze(0),
                    versions=torch.tensor(seq["versions"]).unsqueeze(0),
                    attention_mask=torch.ones(len(seq["input_ids"]), dtype=torch.bool).unsqueeze(0),
                    rewards=torch.tensor([float(scores[i])]),
                )
                item.update(dict(begin_of_trajectory=torch.tensor([int(first_llm_gen)])))
                item.update({k: torch.tensor([v]) for k, v in traj_stats.items()})
                first_llm_gen = False
                results.append(TensorDict(item, batch_size=[1]))
            trace_payload["trajectories"].append(traj_diag)

        if self.dump_dir is not None:
            os.makedirs(os.path.join(self.dump_dir, str(version)), exist_ok=True)
            jsonl_path = os.path.join(self.dump_dir, str(version), f"{qid}.jsonl")
            with open(jsonl_path, "w") as file_obj:
                for i, (traj_memory, raw_score) in enumerate(zip(traj_memories, raw_scores)):
                    file_obj.write(
                        json.dumps(
                            dict(
                                memory=traj_memory.to_dict(),
                                reward=raw_score,
                                ground_truth=ground_truth,
                                traj_idx=i,
                            )
                        )
                        + "\n"
                    )
            logger.info(
                "Episode dump qid=%s version=%s jsonl_path=%s trajs=%s raw_scores=%s",
                qid,
                version,
                jsonl_path,
                len(traj_memories),
                raw_scores,
            )
        self._dump_episode_trace(version=version, qid=qid, payload=trace_payload)
        logger.info(
            "Episode trace qid=%s version=%s trace_dir=%s normalized_scores=%s",
            qid,
            version,
            self.dump_dir,
            scores,
        )

        return concat_padded_tensors([dict(td) for td in results])


@dataclass
class AgentLightRLConfig(GRPOConfig):
    async_training: bool = field(
        default=True, metadata={"help": "Whether to decouple rollout from training"}
    )
    max_turns: int = field(default=12, metadata={"help": "Maximum number of turns"})
    n_trajs: int = field(default=4, metadata={"help": "Trajectories per query"})
    search_only: bool = field(
        default=False,
        metadata={"help": "If true, use the search-only prompt instead of search+access"},
    )
    search_client_type: str = field(
        default="async-search-access",
        metadata={"help": "Tool client type"},
    )
    reward_type: str = field(default="F1", metadata={"help": "Reward function"})
    topk: int = field(default=3, metadata={"help": "Top-k search results"})
    valid_inst_ratio: float = field(default=1.0, metadata={"help": "Valid instruction ratio"})
    log_agent_stats: bool = field(default=True, metadata={"help": "Log agent stats"})
    log_agent_stats_keys: List[str] = field(
        default_factory=lambda: [
            "num_input_tokens",
            "num_output_tokens",
            "num_llm_gens",
            "num_search_queries",
            "num_success_search_queries",
            "num_failed_search_queries",
            "num_pages",
            "num_success_url_accesses",
            "num_failed_url_accesses",
            "score",
            "judge_q_invalid",
            "format_reward",
            "repeated_actions",
        ],
        metadata={"help": "Agent stats keys"},
    )
    max_doc_chars: int = field(default=1200, metadata={"help": "Chars kept per search doc"})
    max_page_total_chars: int = field(
        default=30000, metadata={"help": "Total webpage chars kept"}
    )
    max_page_chunk_chars: int = field(
        default=6000, metadata={"help": "Chars kept per webpage chunk"}
    )
    max_page_chunks: int = field(default=3, metadata={"help": "Maximum webpage chunks"})
    short_page_preview_chars: int = field(
        default=160, metadata={"help": "Short preview chars for memory"}
    )
    use_short_context_for_rollout: bool = field(
        default=True, metadata={"help": "Use short summaries instead of full pages during rollout"}
    )
    search_step_penalty: float = field(
        default=0.01, metadata={"help": "Penalty per search call"}
    )
    access_step_penalty: float = field(
        default=0.02, metadata={"help": "Penalty per access call"}
    )
    repeated_action_penalty: float = field(
        default=0.02, metadata={"help": "Penalty for repeated actions"}
    )
    no_evidence_penalty: float = field(
        default=0.2, metadata={"help": "Penalty when the agent answers without any search/access"}
    )
    no_access_penalty: float = field(
        default=0.1,
        metadata={"help": "Penalty when the agent searches but answers without opening a result URL"},
    )


def get_search_dataset(dataset_path, rank, world_size):
    dataset = load_dataset(path="json", split="train", data_files=dataset_path)
    return split_dataset_by_node(dataset, rank=rank, world_size=world_size)


def main(args):
    config, _ = load_expr_config(args, AgentLightRLConfig)
    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    allocation_mode = _AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train

    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)

    train_dataloader = StatefulDataLoader(
        get_search_dataset(config.train_dataset.path, rank, world_size),
        batch_size=config.train_dataset.batch_size // world_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=config.train_dataset.drop_last,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)
    actor.initialize(None, ft_spec)

    weight_update_meta = WeightUpdateMeta.from_disk(
        config.experiment_name,
        config.trial_name,
        config.cluster.fileroot,
    )
    actor.connect_engine(rollout, weight_update_meta)

    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)

    workflow = ASearcherLightWorkflow(
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        dump_dir=os.path.join(StatsLogger.get_log_path(config.stats_logger), "generated"),
        dataset_path=config.train_dataset.path,
        max_turns=config.max_turns,
        n_trajs=config.n_trajs,
        search_client_type=config.search_client_type,
        reward_type=config.reward_type,
        topk=config.topk,
        valid_inst_ratio=config.valid_inst_ratio,
        max_tokens=config.actor.mb_spec.max_tokens_per_mb,
        search_only=config.search_only,
        max_doc_chars=config.max_doc_chars,
        max_page_total_chars=config.max_page_total_chars,
        max_page_chunk_chars=config.max_page_chunk_chars,
        max_page_chunks=config.max_page_chunks,
        short_page_preview_chars=config.short_page_preview_chars,
        use_short_context_for_rollout=config.use_short_context_for_rollout,
        search_step_penalty=config.search_step_penalty,
        access_step_penalty=config.access_step_penalty,
        repeated_action_penalty=config.repeated_action_penalty,
        no_evidence_penalty=config.no_evidence_penalty,
        no_access_penalty=config.no_access_penalty,
    )

    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)
    recover_handler = RecoverHandler(config.recover, ft_spec)
    recover_info = recover_handler.load(
        actor,
        saver,
        evaluator,
        stats_logger,
        train_dataloader,
        inference_engine=rollout,
        weight_update_meta=weight_update_meta,
    )
    start_step = recover_info.last_step_info.next().global_step if recover_info is not None else 0

    total_epochs = config.total_train_epochs
    steps_per_epoch = len(train_dataloader)
    max_steps = total_epochs * steps_per_epoch
    data_generator = cycle_dataloader(train_dataloader)

    for global_step in range(start_step, max_steps):
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch
        step_info = StepInfo(
            global_step=global_step,
            epoch=epoch,
            epoch_step=step,
            steps_per_epoch=steps_per_epoch,
        )

        print(f"Epoch {epoch}. Step: {step}/{steps_per_epoch}")

        with stats_tracker.record_timing("rollout"):
            if config.async_training:
                batch = rollout.prepare_batch(train_dataloader, workflow=workflow)
            else:
                try:
                    data = next(data_generator)
                except StopIteration:
                    data_generator = iter(train_dataloader)
                    data = next(data_generator)
                batch = rollout.rollout_batch(data, workflow=workflow)
            batch = tensor_container_to(batch, actor.device)
            batch = broadcast_tensor_container(
                batch,
                src_rank=actor.current_data_parallel_head(),
                group=actor.context_and_model_parallel_group,
            )

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
            with stats_tracker.record_timing("recompute_logp"):
                prox_logps = actor.compute_logp(batch)
                for traj, logp in zip(batch, prox_logps):
                    traj["prox_logp"] = logp
                log_gpu_stats("recompute logp")

        with stats_tracker.record_timing("compute_advantage"):
            batch = actor.compute_advantages(batch)
            log_gpu_stats("compute advantages")

        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

        with (
            stats_tracker.record_timing("train_step"),
            stats_tracker.scope("grpo_actor"),
        ):
            actor.ppo_update(batch)
            actor.step_lr_scheduler()
            log_gpu_stats("actor update")

        rollout.pause()

        with stats_tracker.record_timing("update_weights"):
            new_version = global_step + 1
            versioned_meta = weight_update_meta.with_version(new_version)
            actor.update_weights(versioned_meta)
            dist.barrier(device_ids=[actor.device.index])
            current_platform.synchronize()
            actor.set_version(new_version)
            rollout.set_version(new_version)

        with stats_tracker.record_timing("save"):
            saver.save(actor, epoch, step, global_step, tokenizer=tokenizer)

        with stats_tracker.record_timing("checkpoint_for_recover"):
            recover_handler.dump(
                actor,
                step_info,
                saver,
                evaluator,
                stats_logger,
                train_dataloader,
            )

        stats = [stats_tracker.export_all(reduce_group=actor.data_parallel_group)]
        stats_logger.commit(epoch, step, global_step, stats)
        evaluator.evaluate(lambda: None, epoch, step, global_step)
        rollout.resume()

    stats_logger.close()
    rollout.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(os.sys.argv[1:])
