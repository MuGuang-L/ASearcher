import queue
import json
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


@dataclass
class Record:
    type: str
    text: str
    token_ids: List[int]
    short_text: str = ""
    input_len: Optional[int] = None
    input_tokens: Optional[List[int]] = None
    output_len: Optional[int] = None
    full_token_ids: Optional[List[int]] = None
    output_tokens: Optional[List[int]] = None
    output_logprobs: Optional[List[float]] = None
    output_versions: Optional[List[int]] = None

    def to_dict(self):
        return asdict(self)


class AgentMemory:
    def __init__(self, prompt, prompt_token_ids):
        self.memory = [Record(type="prompt", text=prompt, token_ids=prompt_token_ids)]

    def llm_gen_count(self):
        return sum(r.type == "llm_gen" for r in self.memory)

    def filter_records(self, record_type):
        return [r for r in self.memory if r.type == record_type]

    def prepare_prompt(self):
        prompt = ""
        for record in self.memory:
            if record.type == "prompt":
                prompt = record.text
            elif record.type in ["search_results", "webpage"]:
                prompt = prompt + "\n\n" + record.short_text + "\n<think>\n"
            elif record.type == "llm_gen":
                prompt = prompt + record.text
            else:
                raise RuntimeError(f"Unknown record type: {record.type}")
        return prompt

    def prepare_prompt_token_ids(self):
        prompt_token_ids = []
        for record in self.memory:
            prompt_token_ids += record.token_ids
        return prompt_token_ids

    def add_record(self, record: Record):
        self.memory.append(record)

    def logging_stats(self) -> Dict:
        llm_gens = self.filter_records(record_type="llm_gen")
        search_results = self.filter_records(record_type="search_results")
        webpages = self.filter_records(record_type="webpage")
        return dict(
            num_llm_gens=len(llm_gens),
            num_input_tokens=sum(len(r.input_tokens or []) for r in llm_gens),
            num_output_tokens=sum(len(r.output_tokens or []) for r in llm_gens),
            num_search_queries=len(search_results),
            num_success_search_queries=len(
                [r for r in search_results if "No search results are found" not in r.text]
            ),
            num_failed_search_queries=len(
                [r for r in search_results if "No search results are found" in r.text]
            ),
            num_pages=len(webpages),
            num_success_url_accesses=len(
                [r for r in webpages if ">>>> Page 1 >>>>" in r.text]
            ),
            num_failed_url_accesses=len(
                [r for r in webpages if ">>>> Page 1 >>>>" not in r.text]
            ),
        )

    def to_dict(self):
        return [record.to_dict() for record in self.memory]


class SearchAgentLight:
    def __init__(
        self,
        prompt,
        prompt_token_ids,
        *,
        max_doc_chars: int = 1200,
        max_page_total_chars: int = 30000,
        max_page_chunk_chars: int = 6000,
        max_page_chunks: int = 3,
        short_page_preview_chars: int = 160,
        use_short_context_for_rollout: bool = True,
    ):
        self.prompt = prompt
        self.memory = AgentMemory(prompt=prompt, prompt_token_ids=prompt_token_ids)
        self.summary_job_queue = queue.Queue(64)
        self.max_doc_chars = max_doc_chars
        self.max_page_total_chars = max_page_total_chars
        self.max_page_chunk_chars = max_page_chunk_chars
        self.max_page_chunks = max_page_chunks
        self.short_page_preview_chars = short_page_preview_chars
        self.use_short_context_for_rollout = use_short_context_for_rollout

    @property
    def num_turns(self):
        return self.memory.llm_gen_count()

    @property
    def is_finished(self):
        pattern = r"<answer>(.*?)</answer>"
        return any(
            len(re.findall(pattern, record.text, re.DOTALL)) > 0
            for record in self.memory.filter_records("llm_gen")
        )

    def add_summary_jobs(self, summary_jobs):
        if not isinstance(summary_jobs, list):
            summary_jobs = [summary_jobs]
        for summary_job in summary_jobs:
            assert summary_job.get("type", "unknown") in [
                "search_results",
                "webpage",
            ], "Unknown summary job type"
            self.summary_job_queue.put_nowait(summary_job)

    def prepare_llm_query(self, tokenizer):
        prompt_token_ids = self.memory.prepare_prompt_token_ids()
        sampling_params = dict(stop=["</tool_call>", "</search>", "</access>", "</answer>"])
        if not self.summary_job_queue.empty():
            summary_job = self.summary_job_queue.get_nowait()
            if summary_job["type"] in ["search_results", "webpage"]:
                full_text = "\n\n" + summary_job["text"] + "\n<think>\n"
                short_text = (
                    "\n\n"
                    + summary_job.get("short_text", summary_job["text"])
                    + "\n<think>\n"
                )
                full_token_ids, short_token_ids = tokenizer(
                    [full_text, short_text], add_special_tokens=False
                )["input_ids"]
                record = Record(
                    type=summary_job["type"],
                    text=full_text,
                    short_text=short_text,
                    token_ids=short_token_ids,
                    full_token_ids=full_token_ids,
                )
                prompt_token_ids += (
                    short_token_ids if self.use_short_context_for_rollout else full_token_ids
                )
                self.memory.add_record(record)
                sampling_params["stop"] = ["</think>"]
        return prompt_token_ids, sampling_params

    def consume_llm_response(self, resp, completion_text):
        record = Record(
            type="llm_gen",
            text=completion_text,
            token_ids=resp.output_tokens,
            input_len=resp.input_len,
            input_tokens=resp.input_tokens,
            output_len=resp.output_len,
            output_tokens=resp.output_tokens,
            output_logprobs=resp.output_logprobs,
            output_versions=resp.output_versions,
        )
        self.memory.add_record(record)

        tool_calls = []

        tool_call_matches = re.findall(r"<tool_call>(.*?)</tool_call>", completion_text, re.DOTALL)
        for match in tool_call_matches:
            normalized = self._normalize_structured_tool_call(match)
            if normalized is not None:
                tool_calls.append(normalized)

        for pattern in [
            r"<search>(.*?)</search>",
            r"<access>(.*?)</access>",
            r"<answer>(.*?)</answer>",
        ]:
            matches = re.findall(pattern, completion_text, re.DOTALL)
            if matches:
                candidate = str(pattern.replace("(.*?)", matches[-1]))
                tool_calls.append(self._normalize_legacy_tool_call(candidate))
        return tool_calls

    def _normalize_structured_tool_call(self, payload: str):
        try:
            call = json.loads(payload.strip())
        except Exception:
            return None

        name = str(call.get("name", "")).strip().lower()
        arguments = call.get("arguments", {}) or {}
        if name == "search":
            query = arguments.get("query")
            if isinstance(query, list):
                query = next((q for q in query if isinstance(q, str) and q.strip()), None)
            if isinstance(query, str) and query.strip():
                return self._normalize_legacy_tool_call(f"<search>{query.strip()}</search>")
        if name in {"visit", "access"}:
            url = arguments.get("url")
            if isinstance(url, list):
                url = next((u for u in url if isinstance(u, str) and u.strip()), None)
            if isinstance(url, str) and url.strip():
                return self._normalize_legacy_tool_call(f"<access>{url.strip()}</access>")
        return None

    def _normalize_legacy_tool_call(self, tool_call: str):
        if tool_call.startswith("<search>") and tool_call.endswith("</search>"):
            query = tool_call[len("<search>") : -len("</search>")].strip()
            if re.match(r"^https?://", query):
                return f"<access>{query}</access>"
            return f"<search>{query}</search>"
        if tool_call.startswith("<access>") and tool_call.endswith("</access>"):
            url = tool_call[len("<access>") : -len("</access>")].strip()
            return f"<access>{url}</access>"
        if tool_call.startswith("<answer>") and tool_call.endswith("</answer>"):
            answer = tool_call[len("<answer>") : -len("</answer>")].strip()
            return f"<answer>{answer}</answer>"
        return tool_call

    def consume_tool_response(self, res, topk=3):
        if res["type"] == "search":
            summary_job = dict(type="search_results")
            documents = res["documents"][:topk]
            urls = res["urls"][:topk]

            if documents:
                doc_id_template = "[Doc {doc_id}]({url}):\n"
                access_hint = (
                    "\n\nIf a snippet is insufficient, open one of these exact URLs with "
                    "<access> url </access> or "
                    '<tool_call>{"name": "visit", "arguments": {"url": "exact_url"}}</tool_call>.'
                )
                text = (
                    "<information>\n"
                    + "\n\n".join(
                        [
                            doc_id_template.format(doc_id=str(i + 1), url=url)
                            + doc[: self.max_doc_chars]
                            for i, (doc, url) in enumerate(zip(documents, urls))
                        ]
                    )
                    + access_hint
                    + "\n</information>"
                )
            else:
                text = "<information>\nNo search results are found.\n</information>"
            summary_job["text"] = text
            self.add_summary_jobs(summary_job)

        elif res["type"] == "access":
            summary_jobs = []
            page = res["page"]
            if page is not None and page.strip() != "":
                page = page[: self.max_page_total_chars]
                while len(page) > 0 and len(summary_jobs) < self.max_page_chunks:
                    chunk_len = min(self.max_page_chunk_chars, len(page))
                    chunk = page[:chunk_len]
                    summary_jobs.append(
                        dict(
                            type="webpage",
                            text=(
                                "<information>\n>>>> Page "
                                + str(len(summary_jobs) + 1)
                                + " >>>>\n\n"
                                + chunk
                                + "\n</information>"
                            ),
                            short_text=(
                                "<information>\n>>>> Page "
                                + str(len(summary_jobs) + 1)
                                + " >>>>\n\n"
                                + chunk[: self.short_page_preview_chars]
                                + "\n</information>"
                            ),
                        )
                    )
                    page = page[chunk_len:]
            else:
                summary_jobs.append(
                    dict(
                        type="webpage",
                        text="<information>\nNo More Information is Found for this URL.\n</information>",
                    )
                )
            self.add_summary_jobs(summary_jobs)

    def get_answer(self):
        text = self.memory.prepare_prompt()
        pattern = r"<answer>(.*?)</answer>"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None
