# ASearcher 接手分析

本文档按“我就是接手者”的视角记录当前仓库状态、可运行路径、主要风险和接手后的执行顺序。

## 1. 我接手到的是什么

这不是一个干净的单一项目，而是三层叠加：

1. 上游 `ASearcher` 搜索智能体训练项目
2. 仓库内 vendored 的 `AReaL` 训练框架
3. 后续新增的 lightweight 本地训练路径、Docker 工作流和 trace viewer

当前最实用的入口不是原始大规模 web RL 训练，而是轻量本地路径：

- `scripts/run_light_local.sh`
- `ASearcher/train/asearcher_light.py`
- `ASearcher/train/search_agent_light.py`
- `ASearcher/configs/asearcher_local_light_qwen3.yaml`

结论：这个仓库现在更像“基于 ASearcher/AReaL 的可修改实验平台”，而不是一个已经收敛成型的产品仓库。

## 2. 我确认到的主链路

### 2.1 本地轻量训练链路

运行顺序是：

1. `scripts/build_index.sh`
2. `scripts/launch_local_server.sh`
3. `scripts/run_light_local.sh`

职责拆分：

- `build_index.sh`：用 `utils/index_builder.py` 为本地 wiki 语料构建向量索引
- `launch_local_server.sh`：启动 `tools/local_retrieval_server.py`
- `run_light_local.sh`：设置 `PYTHONPATH` 和 `RAG_SERVER_ADDR_DIR`，然后通过 `areal.launcher.local` 启动 `ASearcher/train/asearcher_light.py`

### 2.2 训练时的核心调用链

`ASearcher/train/asearcher_light.py` 是训练总入口，主要做四件事：

1. 加载 `AgentLightRLConfig`
2. 初始化 `FSDPPPOActor` 和 `RemoteSGLangEngine`
3. 用 `ASearcherLightWorkflow` 生成 rollout
4. 用 PPO/GRPO 风格流程做更新、保存、recover、stats logging

`ASearcherLightWorkflow` 内部逻辑：

1. 对单个样本构造 prompt
2. 并行采样 `n_trajs`
3. 每条轨迹通过 `SearchAgentLight` 执行多轮 `<search>` / `<access>` / `<answer>`
4. 工具调用交给 `SearchToolBox`
5. 根据答案抽取结果和格式奖励计算 score
6. 对多轨迹做均值中心化，转成训练 batch
7. 同时把原始 rollout 和 `*.trace.json` 落盘

### 2.3 agent 状态机

`ASearcher/train/search_agent_light.py` 是一个很关键的“记忆 + 状态机”文件：

- `AgentMemory` 维护 prompt、search results、webpage、llm_gen
- `prepare_llm_query()` 把记忆拼成下一轮输入
- `consume_llm_response()` 从模型输出中抽 `<search>`、`<access>`、`<answer>`
- `consume_tool_response()` 把搜索结果或网页内容写回 memory

这条 lightweight 路径的本质不是改训练器，而是把上下文压缩得更便宜：

- 搜索摘要更短
- 网页截断更激进
- chunk 更少
- rollout 可只保留短摘要上下文

## 3. 工具层和 reward 是怎么接上的

`ASearcher/utils/search_tool.py` 是训练工作流和搜索后端之间的桥：

- 读取训练数据，建立 `qid -> answer` 映射
- 通过 `make_search_client()` 创建搜索客户端
- 执行 `<search>` 时返回 `documents` / `urls`
- 执行 `<access>` 时返回 `page`
- 每一步都顺手从 action 中抽 `<answer>` 并算 reward

这意味着 reward 并不是 episode 结束后统一算，而是“每次动作后都尝试从当前 action 里抽答案并打分”，最终轨迹分数由：

- 基础答案分
- 格式奖励
- search/access/repeated action penalty

共同构成。

这是当前代码的重要设计点，也是后续最容易改 reward shaping 的地方。

## 4. 搜索后端有两套

### 4.1 本地检索

`async-search-access`

对应 `ASearcher/utils/search_utils.py` 里的 `AsyncSearchBrowserClient`。

它依赖：

- `RAG_SERVER_ADDR_DIR`
- 本地 RAG server 写入的 `Host*_IP*.txt`

适合 lightweight 本地训练。

### 4.2 在线搜索

`async-online-search-access`

对应 `AsyncOnlineSearchClient`，依赖：

- `SERPER_API_KEY`
- 可选 `JINA_API_KEY`

适合 web search 训练和评测。

结论：本地轻量路径和线上 web 路径在 agent 逻辑上相近，但在 infra 依赖上是两套系统。

## 5. 评测链路

评测主入口是：

- `evaluation/search_eval_async.py`

它和训练路径基本解耦，职责是：

- 加载 benchmark 数据
- 用 `agent/` 下的 agent 实现跑推理
- 用 `tools/search_utils.py` 接搜索后端
- 聚合 F1 / EM / CEM / LLM-as-Judge 结果

要注意：

1. 评测目录下依赖的是顶层 `agent/`、`tools/`
2. 训练目录下依赖的是 `ASearcher/` 包内实现

也就是说，仓库里存在一套“训练版实现”和一套“评测/演示版实现”的并存结构，后续维护有重复成本。

## 6. demo / 可观测性

现有两个可视化方向：

1. `demo/asearcher_demo.py`：传统网页 demo
2. `demo/light_trace_server.py`：轻量训练 trace 查看器

我认为当前真正有价值的是 trace viewer，因为它直接服务训练调试：

- episode 状态
- 轨迹得分
- penalty breakdown
- 每一步 tool call 和 tool result preview
- 最终答案

如果我要继续接手推进，trace viewer 比旧 demo 更接近当前主线。

## 7. 现有 handoff 文档给我的真实信息

已有两份 handoff：

- `HANDOFF_ASEARCHER_LIGHT.md`
- `HANDOFF_DOCKER_AND_NEXT_STEPS.md`

我从里面提炼出的有效结论是：

1. 当前维护者已经把 lightweight 路径和 trace 观测补上了
2. Docker 运行时、NVIDIA runtime、代理透传都做过一轮修复
3. 当前推荐开发环境是容器内 `/workspace/ASearcher`
4. 一个尚未消除的关键不确定性是：运行时到底 import 的是镜像里预装的 `areal`，还是仓库内 vendored 的 `AReaL`

这个不确定性很重要，因为：

- 如果 import 的不是仓库内源码
- 我改 `AReaL` 不一定生效
- 调试训练器会出现“代码看着对，运行却没变”的假象

## 8. 我接手后确认到的风险点

### 8.1 仓库不是单一真相源

同类逻辑分散在：

- `ASearcher/*`
- 顶层 `agent/*`
- 顶层 `tools/*`
- vendored `AReaL/*`

这会导致修改一处、另一处行为不一致。

### 8.2 路径假设仍然偏硬编码

README 和配置大量假设：

- `/home/ubuntu/ASearcher`
- `/workspace/ASearcher`
- `/tmp/areal/...`

这对单机交接还行，但不利于后续迁移和自动化。

### 8.3 `git` 当前不可直接用

我实际检查时，`git status` 返回：

- `fatal: detected dubious ownership in repository at '/workspace/ASearcher'`

说明当前用户和仓库属主不一致，接手后如果要正常用 git，需要先处理 `safe.directory` 或改属主。

### 8.4 AReaL 来源可能有二义性

handoff 已明确提醒：

- 容器里可能优先 import 镜像自带 `areal`
- 而不是 `/workspace/ASearcher/AReaL`

这在接手阶段属于高优先级确认项。

### 8.5 轻量路径虽然是主线，但还缺“完整验通”

从现有文档看，已经完成的是：

- 代码新增
- 基本语法检查
- Docker / GPU / 代理 / import 的部分验证

但还没有看到“从建索引到本地 server 到真实训练 step 的完整闭环验收记录”。

## 9. 我对当前项目状态的判断

如果用工程成熟度来描述：

- 算法思路和大规模训练方向是清楚的
- 轻量实验路径已经具备雏形
- 基础设施在逐步收敛
- 但仓库还处于“能迭代、未收口”的过渡态

因此我接手后的策略不应该是先重构，而应该是：

1. 先把一条最短闭环跑通
2. 再确认 import / 配置 / 路径 真正可控
3. 最后再考虑消重和结构治理

## 10. 我接手后的优先执行顺序

### P0

1. 确认容器内 `import areal` 的实际来源
2. 确认 lightweight 配置能否完整加载
3. 确认本地 RAG server 能启动并被训练脚本发现
4. 做一次最小训练 smoke run

### P1

1. 补一个仓库自带 smoke test 脚本
2. 把关键环境变量和路径收敛成统一入口
3. 记录训练输出目录、trace 目录、checkpoint 目录

### P2

1. 处理 `ASearcher/` 与顶层 `agent/`、`tools/` 的重复实现
2. 决定 `AReaL` 长期是 vendored 目录还是 pinned submodule
3. 为 reward shaping、tool backend、trace schema 建稳定接口

## 11. 我现在会把哪个路径当主线

我不会把以下内容当接手第一优先级：

- 16 节点大规模 web RL
- QwQ-32B reasoning 训练
- 旧 demo 页面

我会把下面这条当主线：

1. 容器内开发
2. 本地知识库检索
3. `asearcher_light.py`
4. `Qwen3-1.7B` 轻量配置
5. trace viewer 观察 rollout

原因很简单：这条链路依赖最少、成本最低、反馈最快，最适合作为接手后的稳定基线。

## 12. 一句话总结

我接手到的是一个“以 ASearcher 为原型、以 AReaL 为底座、以 lightweight 本地训练路径为当前最可落地入口”的过渡中仓库。短期目标不是重构成漂亮结构，而是先把轻量闭环稳定跑通，并消除 `areal` 来源、路径假设和重复实现带来的维护风险。
