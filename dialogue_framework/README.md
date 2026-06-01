# Dialogue Framework — LangGraph + SqliteSaver 版

基于 LangGraph 把"债券机器人多轮对话"建模成一张完整的状态机有向图。
checkpointer 自动用 SQLite 按 session_id 持久化对话状态,
**所以追问、槽位累积、确认、取消全部自动跨轮工作**。

## 跑通方式

本项目支持两种用法 —— **意图识别评测** 和 **多轮对话演示**。

### 准备
```bash
pip install -r requirements.txt
# 把语料 Excel 放进项目目录(跟 config.py 同级)
# 文件名:FICC机器人优化迭代清单.xlsx
```

### 用法 1:意图识别评测(跟 intent_framework 等价)

跑全量评测,**输入输出格式跟之前 intent_framework 完全一样**:
```bash
python eval_intent.py                          # 全量评测
python eval_intent.py --limit 20               # 只跑前 20 条(冒烟测试)
python eval_intent.py --dry-run                # 不调 LLM,只验证数据加载
python eval_intent.py --output-tag baseline    # 给输出文件加标签
```
输出在 `outputs/` 目录:
- `<tag>_report.txt` —— 准确率/F1/混淆矩阵/延迟分位数
- `<tag>_results.csv` —— 全量预测明细
- `<tag>_bad_cases.csv` —— 错误 case(看这个改 prompt)

**实现说明**:eval_intent.py 复用了 dialogue_framework 的 `build_classify_messages` 和 `llm_client`,只是评测时传空 state(stage=init),走"无上下文"分支,等价于单轮分类。这样:
- prompt 就是 dialogue_framework 里更新过的那版(包含多轮规则,但单轮模式下不影响)
- 评测出来的指标可直接和之前 intent_framework 的 baseline 对比
- 改 prompts.py 优化,**评测和对话效果同时改进**,不用维护两份 prompt

### 用法 2:多轮对话演示(展示给领导/同事)

```bash
python main.py                  # 跑 6 个内置场景
python main.py --scenario 1     # 只跑第 1 个场景(多轮追问)
python main.py --interactive    # 进交互模式,自己输入
```

展示效果:每轮都会显式打出
- 用户输入、系统回复
- 当前意图、当前阶段、当前槽位(灰色,给技术同事看)
- **🔧 API 调用追踪**(紫色,展示"系统正确路由到了哪个 API、传了什么参数、API 返回什么")

由于尚无真实交易 API 可接,`api_adapters.py` 里的方法都是**占位实现**(返回模拟的订单号、订单列表等)。但**整条调用链路是真实的** —— LLM 真打、状态机真跑、API 真按契约被调用,只是 API 内部的实现是 stub。接入真实 API 时,只需替换 `api_adapters.py` 里每个方法内部的实现,上层代码不动。

## 完整有向图

```
                            START
                              │
                              ▼
                      ┌───────────────┐
                      │   classify    │  ← LLM 调用 #1(意图+槽位)
                      └───────┬───────┘
                              │
                         <route_action>
        ┌────────────┬─────────────┬──────────┬─────────────────┐
        │            │             │          │                 │
        ▼            ▼             ▼          ▼                 ▼
  ┌──────────┐ ┌──────────┐  ┌──────────┐ ┌──────────┐  ┌────────────────┐
  │chat_reply│ │ask_miss- │  │   ask    │ │ execute  │  │ handle_confirm │
  │          │ │ing_slots │  │ _confirm │ │ _query / │  │  (前一轮在     │
  │ LLM #2   │ │ (模板)   │  │  (模板)  │ │ selection│  │   confirming)  │
  │          │ │          │  │          │ │ / cancel │  │                │
  └────┬─────┘ └────┬─────┘  └────┬─────┘ └────┬─────┘  └────────┬───────┘
       │            │             │            │                  │
       │            │             │            │             <user_yes_no>
       │            │             │            │              ┌───┴────┐
       │            │             │            │              ▼        ▼
       │            │             │            │         execute_   reset
       │            │             │            │         inquiry    (回 init)
       │            │             │            │              │        │
       ▼            ▼             ▼            ▼              ▼        ▼
                                  END
```

**每次 `graph.invoke(user_text, thread_id)` 跑一次这张图。**
中间路径不同,经过 1-3 个节点后到达 END,本轮结束。
下一轮再调 invoke 时,checkpointer 把上轮的 state(slots、stage 等)自动加载回来。

## 项目目录

```
dialogue_framework/
├── config.py            # API key、模型、SQLite 路径、mock 开关、Excel 路径、SHEET_CONFIGS
├── state.py             # DialogueState TypedDict 定义 + 槽位检查工具
├── prompts.py           # 所有 prompt 模板(关键:把 state 转 LLM 上下文)
├── llm_client.py        # LLM 调用封装(支持 mock / 真实两套)
├── api_adapters.py      # 外部交易 API 的 mock,真接入时改这里 + 演示用调用追踪
├── nodes.py             # ★ 所有图节点的实现(项目最核心文件)
├── routes.py            # 条件边路由函数
├── graph.py             # 组装图 + handle_turn 主入口
├── data_loader.py       # 从 xlsx 加载语料(给 eval_intent.py 用)
├── eval_intent.py       # ★ 意图识别评测脚本(跟 intent_framework 等价)
├── main.py              # ★ 对话演示入口(美化输出 + API 调用追踪)
├── requirements.txt
└── README.md
```

## 关键概念:LangGraph 怎么"管"上下文

**LangGraph 不自动管 LLM messages,它管的是状态机的 state 流转。**

把它当成两层:

| 层 | 谁管 | 怎么管 |
|---|---|---|
| **state 流转**(slots、stage、recent_turns 等字段跨轮持久化) | LangGraph + SqliteSaver | 全自动,按 thread_id 存取 |
| **LLM 上下文**(每次 chat completions 调用的 messages 数组) | 你在 node 函数里写代码 | `build_xxx_messages(state)` 把 state 拼成 system+few-shot+user |

具体看 `prompts.py` 里 `build_classify_messages`:

```python
# system + few-shot 完全稳定 → KV cache 可复用 → 延迟低
messages = [{"role": "system", "content": CLASSIFY_SYSTEM_PROMPT}]
for ex_in, ex_out in CLASSIFY_FEW_SHOTS:
    messages.append({"role": "user", "content": ex_in})
    messages.append({"role": "assistant", "content": ex_out})

# 只有最后这条 user message 每轮变化
# 把 state 里的 slots / stage / last_question 转成"结构化上下文文本"
user_msg = f"""当前会话上下文:
- 阶段: {state['stage']}
- 已收集槽位: {json.dumps(state['slots'])}
- 上轮系统问: {state['last_question']}

用户本轮输入:{user_text}"""
messages.append({"role": "user", "content": user_msg})
```

模型每次看到的:**前面一大段固定的指令和示例,只有最后一段结构化状态在变**。
这样模型既能理解"我处于多轮对话的哪个阶段",又不会因为前缀变化让 KV cache 失效。

## State 设计(state.py)

```python
class DialogueState(TypedDict):
    user_text: str          # 本轮输入
    intent: str             # 意图分类结果
    slots: dict             # 槽位(跨轮累积)
    stage: str              # 流程阶段:init/collecting_slots/confirming/executed/chatting
    reply: str              # 本轮回复
    last_question: str      # 系统刚刚问的(下一轮喂给模型的上下文)
    recent_turns: list      # 最近几轮原文,带 add reducer 实现追加
    session_id: str
    turn_count: int
```

注意 `recent_turns: Annotated[list, operator.add]`:这告诉 LangGraph
"node 返回这个字段时不是覆盖,而是 append"。其他字段默认是覆盖语义。

## 节点(nodes.py)里发生了什么

每个 node 是个函数,签名都是 `def xxx_node(state) -> dict`。

- **输入**:LangGraph 自动把完整 state 喂进来(已经从 SQLite 加载)
- **输出**:dict,只包含本节点想改的字段,LangGraph 会合并到 state

举例 `classify_node`:
```python
def classify_node(state):
    user_text = state["user_text"]
    # 把 state 转成 LLM 上下文(就在这里!)
    messages = prompts.build_classify_messages(user_text, state)
    result = llm_client.chat(messages)
    parsed = parse(result["content"])
    
    # 槽位累积合并(新值非 null 才覆盖旧值)
    merged_slots = _merge_slots(state["slots"], parsed["slots"])
    
    return {
        "intent": parsed["intent"],
        "slots": merged_slots,           # 覆盖整个 slots(因为 dict 字段是覆盖语义)
        "turn_count": state["turn_count"] + 1,
        "recent_turns": [{"role":"user", "text":user_text}],  # 这条会被 append
    }
```

## 跨轮持久化:checkpointer 在做什么

`graph.py` 里:
```python
checkpointer = SqliteSaver(sqlite3.connect("dialogue_checkpoints.db"))
graph = builder.compile(checkpointer=checkpointer)
```

之后每次 `graph.invoke(inputs, config={"configurable":{"thread_id": session_id}})`:
1. **加载**:checkpointer 按 thread_id 从 SQLite 读最新 state(若是新会话则为空)
2. **合并**:把你传的 `inputs`(通常只有 `{"user_text": "..."}`)合并进 state
3. **执行**:跑图,每个 node 看到的是完整 state
4. **保存**:跑完后把新 state 写回 SQLite

效果:**你只在外部代码里关心"用户说了什么",其他全交给框架**。

## 怎么接你的真实 LLM 和 API

- **LLM 端**:已默认走真实 endpoint。要换模型/账号,只改 `config.py` 里的 `API_KEY / BASE_URL / MODEL_ID`。
- **交易 API 端**:`api_adapters.py` 里 4 个类(InquiryAPI / CancelAPI / QueryAPI / SelectionAPI),当前是占位实现。接入真实系统时,把每个方法里 `# TODO` 标注那段换成真实 HTTP/RPC 调用即可。接口契约(参数+返回字段)保持不变,上层代码无需修改。

## 跟之前 intent_framework 的关系

- **intent_framework**(意图识别框架)→ 现在变成 `classify_node` 内部 1 个节点
- **dialogue_framework**(本项目)→ 把意图识别上升为多轮对话流程,带状态、追问、确认、调 API

你之前调好的 prompt 直接搬到 `prompts.py` 的 `CLASSIFY_SYSTEM_PROMPT` 即可。

## 还可以怎么扩展

1. **意图切换** — 用户在 confirming 阶段说了新意图(比如改问"我的订单"),
   目前会被当成"含糊回复",可在 `handle_confirm_node` 里检测出新意图回到 classify。
2. **pending_intents 队列** — 教程提的"用户变更主问题"场景,
   在 state 里加 `pending_intents: list`,中途插入的诉求记入队列。
3. **事件日志** — 在每个 node 入口/出口写一条 event,落 SQLite 单独的表,
   做"事件溯源"。教程推荐的可观测性来源。
4. **转人工节点** — 加 `escalate_node`,出口条件:连续 3 轮 ask_missing 仍未收齐 /
   API 连续失败 / 用户明确说"找人"。
5. **真用 LangGraph 的可视化** — `graph.get_graph().draw_mermaid()` 能输出
   流程图的 mermaid 代码,贴到 mermaid.live 可视化。

## 动态 few-shot + 多级细分(v6)

意图识别现在支持:
- **5 个意图**:bond_inquiry / bond_query / bond_selection / chat / **transfer_human**(转人工)
- **bond_query 细分 order_status**:all / pending / completed
- **bond_inquiry 撤单细分 cancel_scope**:all(全撤) / specified(指定撤)
- **动态 few-shot**:`retriever.py` 用 BAAI/bge-small-zh-v1.5 做 embedding 召回,然后按意图均衡 round-robin 交织(避免最近性偏差)

### 启动前一次性操作
向量库需要先生成(后续启动只读 .npy,无重算):
```bash
# 国外环境(能访问 HuggingFace):
python _build_vectors.py

# 国内环境(推荐用 modelscope 镜像):
pip install modelscope
python _build_vectors.py --source modelscope

# 或者已经下好模型到本地:
python _build_vectors.py --model-path /path/to/bge-small-zh-v1.5
```
生成的 `fewshot_pool_vectors.npy` 约 120KB,提交到 git 也行。


###

实验: E0 基线
USE_HIERARCHICAL_ROUTING: False
RETRIEVAL_MODE: "bge"
USE_VOTING: False
运行命令: D:\anaconda3\envs\care-gnn\python.exe eval_intent.py --output-tag e0_baseline
验证重点: 与 v6 现状一致,作为对比基准
────────────────────────────────────────
实验: E1 分层路由
USE_HIERARCHICAL_ROUTING: True
RETRIEVAL_MODE: "bge"
USE_VOTING: False
运行命令: D:\anaconda3\envs\care-gnn\python.exe eval_intent.py --output-tag e1_hierarchical
验证重点: order_status / cancel_scope 准确率是否提升;延迟是否翻倍
────────────────────────────────────────
实验: E2 多路召回
USE_HIERARCHICAL_ROUTING: False
RETRIEVAL_MODE: "multi_route"
USE_VOTING: False
运行命令: D:\anaconda3\envs\care-gnn\python.exe eval_intent.py --output-tag e2_multi_route
验证重点: 整体 intent 准确率;边缘 case(撤单/转人工)是否改善
────────────────────────────────────────
实验: E3 投票
USE_HIERARCHICAL_ROUTING: False
RETRIEVAL_MODE: "bge"
USE_VOTING: True
运行命令: D:\anaconda3\envs\care-gnn\python.exe eval_intent.py --output-tag e3_voting
验证重点: 准确率小幅提升 vs 延迟约 3× 的 trade-off
────────────────────────────────────────
实验: E4 分层+多路
USE_HIERARCHICAL_ROUTING: True
RETRIEVAL_MODE: "multi_route"
USE_VOTING: False
运行命令: D:\anaconda3\envs\care-gnn\python.exe eval_intent.py --output-tag e4_hier_multi
验证重点: E1+E2 叠加是否有协同效应
────────────────────────────────────────
实验: E5 分层+投票
USE_HIERARCHICAL_ROUTING: True
RETRIEVAL_MODE: "bge"
USE_VOTING: True
运行命令: D:\anaconda3\envs\care-gnn\python.exe eval_intent.py --output-tag e5_hier_vote
验证重点: 子分类准确率上限;延迟约 3-6×
────────────────────────────────────────
实验: E6 全开上限
USE_HIERARCHICAL_ROUTING: True
RETRIEVAL_MODE: "multi_route"
USE_VOTING: True
运行命令: D:\anaconda3\envs\care-gnn\python.exe eval_intent.py --output-tag e6_full
验证重点: 系统准确率天花板;延迟代价
────────────────────────────────────────
实验: E7 召回对照
USE_HIERARCHICAL_ROUTING: False
RETRIEVAL_MODE: "static"
USE_VOTING: False
运行命令: D:\anaconda3\envs\care-gnn\python.exe eval_intent.py --output-tag e7_static
验证重点: 动态召回到底有没有用(与 E0 对比)

---
切换方式(只改 config.py 末尾三行)

# 实验 E1 示例
USE_HIERARCHICAL_ROUTING = True
RETRIEVAL_MODE = "bge"
USE_VOTING = False

---
结果文件位置

每个实验跑完后在 outputs/ 生成三个文件:
┌─────────────────────┬────────────────────────────────────────────┐
│        文件         │                    内容                    │
├─────────────────────┼────────────────────────────────────────────┤
│ <tag>_report.txt    │ 准确率 / F1 / 混淆矩阵 / 延迟 / Token 消耗 │


消融实验操作表

RETRIEVAL_MODE: "bge"
USE_VOTING: False
运行命令: python eval_intent.py --output-tag e0_baseline
验证重点: 与 v6 现状一致,作为对比基准
────────────────────────────────────────
实验: E1 分层路由
USE_HIERARCHICAL_ROUTING: True
RETRIEVAL_MODE: "bge"
USE_VOTING: False
运行命令: python eval_intent.py --output-tag e1_hierarchical
验证重点: order_status / cancel_scope 准确率是否提升;延迟是否翻倍
────────────────────────────────────────
实验: E2 多路召回
USE_HIERARCHICAL_ROUTING: False
RETRIEVAL_MODE: "multi_route"
USE_VOTING: False
运行命令: python eval_intent.py --output-tag e2_multi_route
验证重点: 整体 intent 准确率;边缘 case(撤单/转人工)是否改善
────────────────────────────────────────
实验: E3 投票
USE_HIERARCHICAL_ROUTING: False
RETRIEVAL_MODE: "bge"
USE_VOTING: True
运行命令: python eval_intent.py --output-tag e3_voting
验证重点: 准确率小幅提升 vs 延迟约 3× 的 trade-off
────────────────────────────────────────
实验: E4 分层+多路
USE_HIERARCHICAL_ROUTING: True
RETRIEVAL_MODE: "multi_route"
USE_VOTING: False
运行命令: python eval_intent.py --output-tag e4_hier_multi
验证重点: E1+E2 叠加是否有协同效应
RETRIEVAL_MODE: "bge"
USE_VOTING: True
运行命令: python eval_intent.py --output-tag e5_hier_vote
验证重点: 子分类准确率上限;延迟约 3-6×
────────────────────────────────────────
实验: E6 全开上限
USE_HIERARCHICAL_ROUTING: True
RETRIEVAL_MODE: "multi_route"
USE_VOTING: True
运行命令: python eval_intent.py --output-tag e6_full
验证重点: 系统准确率天花板;延迟代价
────────────────────────────────────────
实验: E7 召回对照
USE_HIERARCHICAL_ROUTING: False
USE_VOTING: False
运行命令: python eval_intent.py --output-tag e7_static
验证重点: 动态召回到底有没有用(与 E0 对比)

---
