"""
所有图节点(node)的实现。

每个 node 是一个函数:
  - 输入:完整的 state(LangGraph 传入)
  - 输出:state 的"更新部分"(dict),只包含本节点要改的字段

在节点里我们做三件事:
  1. 从 state 读出当前上下文(slots、stage、last_question、recent_turns)
  2. 必要时调 LLM(在 prompts.py 里用 build_xxx_messages 组装上下文)
  3. 返回需要更新的字段
"""

import json
import re
from typing import Optional

import llm_client
import prompts
from api_adapters import inquiry_api, query_api, selection_api, cancel_api
from state import (
    DialogueState,
    missing_inquiry_slots,
    all_inquiry_slots_filled,
)
import config


# ============================================================================
# JSON 解析工具(应对 LLM 偶尔包代码块)
# ============================================================================

def _parse_json(text: str) -> Optional[dict]:
    if not text:
        return None
    cleaned = text.strip()
    # 剥离 Qwen3 / DeepSeek-R1 等模型的 thinking 块
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _merge_slots(old: dict, new: dict) -> dict:
    """合并槽位:新值非 None 才会覆盖旧值,这样不会因模型偶尔漏抽把旧槽冲掉。"""
    result = dict(old or {})
    for k, v in (new or {}).items():
        if v is not None and v != "":
            result[k] = v
    return result


# ============================================================================
# 1. classify_node —— 意图识别 + 槽位抽取(入口,每轮必跑)
# ============================================================================

def classify_node(state: DialogueState) -> dict:
    """每轮对话入口:意图识别 + 槽位抽取。
    支持分层路由(USE_HIERARCHICAL_ROUTING)开关。"""
    if config.USE_HIERARCHICAL_ROUTING:
        return _classify_hierarchical(state)
    return _classify_single_shot(state)


def _call_llm_for_classify(messages: list) -> dict:
    """统一 LLM 调用入口:根据 USE_VOTING 决定单次调用或投票。"""
    if config.USE_VOTING:
        return llm_client.vote_chat(messages)
    return llm_client.chat(messages)


def _post_process_classify(state: DialogueState, parsed: dict) -> dict:
    """classify_node 尾部处理:槽位累积、沿用 intent 兜底。"""
    new_intent = parsed.get("intent", "chat")
    new_slots = parsed.get("slots", {}) or {}
    merged = _merge_slots(state.get("slots") or {}, new_slots)

    current_stage = state.get("stage", "init")
    if current_stage in ("collecting_slots", "confirming") and new_intent == "chat":
        new_intent = state.get("intent") or "bond_inquiry"

    return {
        "intent": new_intent,
        "intent_confidence": parsed.get("confidence"),
        "slots": merged,
        "turn_count": state.get("turn_count", 0) + 1,
        "recent_turns": [{"role": "user", "text": state["user_text"]}],
    }


def _classify_single_shot(state: DialogueState) -> dict:
    """单 prompt 方案(baseline,等价于 v6 现有实现)。"""
    user_text = state["user_text"]

    few_shots = None
    if config.USE_DYNAMIC_FEWSHOT and config.RETRIEVAL_MODE != "static":
        from retriever import get_retriever
        few_shots = get_retriever().retrieve(user_text)

    messages = prompts.build_classify_messages(user_text, state, few_shots=few_shots)
    result = _call_llm_for_classify(messages)

    if result["error"]:
        return {"intent": "chat", "reply": "(意图识别异常,请稍后再试)", "next_action": "chat_reply"}

    parsed = _parse_json(result["content"]) or {}
    return _post_process_classify(state, parsed)


def _is_cancel_intent(text: str) -> bool:
    """快速规则判断 bond_inquiry 是否为撤单,不调 LLM。"""
    return "ref" in text.lower() or "撤" in text


def _classify_hierarchical(state: DialogueState) -> dict:
    """分层路由方案:Layer 1 主意图 + 条件性 Layer 2 子分类。"""
    user_text = state["user_text"]

    # ---------- Layer 1:主意图 + 通用槽位 ----------
    few_shots = None
    if config.USE_DYNAMIC_FEWSHOT and config.RETRIEVAL_MODE != "static":
        from retriever import get_retriever
        few_shots = get_retriever().retrieve(user_text)

    l1_messages = prompts.build_l1_messages(user_text, state, few_shots=few_shots)
    l1_result = _call_llm_for_classify(l1_messages)

    if l1_result["error"]:
        return {"intent": "chat", "reply": "(意图识别异常,请稍后再试)", "next_action": "chat_reply"}

    l1_parsed = _parse_json(l1_result["content"]) or {}
    intent = l1_parsed.get("intent", "chat")
    slots = dict(l1_parsed.get("slots") or {})

    # ---------- Layer 2:条件触发 ----------
    # Layer 2A:bond_query → 判 order_status
    if intent == "bond_query":
        l2a_messages = prompts.build_l2_order_status_messages(user_text)
        l2a_result = _call_llm_for_classify(l2a_messages)
        if not l2a_result["error"]:
            l2a_parsed = _parse_json(l2a_result["content"]) or {}
            os_val = l2a_parsed.get("order_status")
            slots["order_status"] = os_val if os_val in config.ORDER_STATUS_LABELS else "all"
        else:
            slots["order_status"] = "all"

    # Layer 2B:bond_inquiry + 撤单 → 判 cancel_scope
    elif intent == "bond_inquiry" and _is_cancel_intent(user_text):
        l2b_messages = prompts.build_l2_cancel_scope_messages(user_text)
        l2b_result = _call_llm_for_classify(l2b_messages)
        if not l2b_result["error"]:
            l2b_parsed = _parse_json(l2b_result["content"]) or {}
            cs_val = l2b_parsed.get("cancel_scope")
            slots["cancel_scope"] = cs_val if cs_val in config.CANCEL_SCOPE_LABELS else "all"
        else:
            slots["cancel_scope"] = "all"

    return _post_process_classify(state, {
        "intent": intent,
        "slots": slots,
        "confidence": l1_parsed.get("confidence", 0.0),
    })


# ============================================================================
# 2. chat_reply_node —— chat 分支,再调一次 LLM 生成自然语言回复
# ============================================================================

def chat_reply_node(state: DialogueState) -> dict:
    """调 LLM 生成 chat 回复。"""
    user_text = state["user_text"]
    messages = prompts.build_chat_messages(user_text, state)
    result = llm_client.chat(messages)
    reply = result["content"] or "(暂时无法回答,请换种方式提问)"

    return {
        "reply": reply,
        "stage": "chatting",
        "last_question": "",
        "recent_turns": [{"role": "assistant", "text": reply}],
    }


# ============================================================================
# 3. ask_missing_slots_node —— 询价缺槽位,生成追问(模板,不调 LLM)
# ============================================================================

# 槽位中文名映射
_SLOT_NAMES_CN = {
    "bond_code": "债券代码", "direction": "方向(bid/ofr)",
    "price": "价格", "amount": "数量", "settle_date": "清算时间",
}


def ask_missing_slots_node(state: DialogueState) -> dict:
    """根据当前缺哪些槽位,生成追问语。模板渲染,不需要 LLM。"""
    missing = missing_inquiry_slots(state.get("slots") or {})
    if not missing:
        # 理论上路由到这里时一定有缺,但保险起见
        reply = "已收集到所有信息。"
        question = ""
    else:
        cn_names = [_SLOT_NAMES_CN.get(s, s) for s in missing]
        reply = f"请补充以下信息:{'、'.join(cn_names)}"
        question = reply

    return {
        "reply": reply,
        "stage": "collecting_slots",
        "last_question": question,
        "recent_turns": [{"role": "assistant", "text": reply}],
    }


# ============================================================================
# 4. ask_confirm_node —— 槽位齐全,生成核对/确认语(模板)
# ============================================================================

def ask_confirm_node(state: DialogueState) -> dict:
    """槽位齐全后,生成核对话术让用户确认。"""
    s = state.get("slots") or {}
    summary = (
        f"请确认下单信息:\n"
        f"  方向: {s.get('direction', '').upper()}\n"
        f"  债券: {s.get('bond_code')}\n"
        f"  价格: {s.get('price')}\n"
        f"  数量: {s.get('amount')}\n"
        f"  清算: {s.get('settle_date', '今+1')}\n"
        f"确认请回复 'y' 或 '确认',取消回复 'n'。"
    )

    return {
        "reply": summary,
        "stage": "confirming",
        "last_question": "请确认是否下单",
        "recent_turns": [{"role": "assistant", "text": summary}],
    }


# ============================================================================
# 5. handle_confirm_node —— 用户在 confirming 阶段的回复处理
# ============================================================================

_YES_PATTERN = re.compile(r"^\s*(y|yes|确认|确定|对|是|ok|好|嗯)\s*$", re.IGNORECASE)
_NO_PATTERN = re.compile(r"^\s*(n|no|取消|不|算了|cancel)\s*$", re.IGNORECASE)


def handle_confirm_node(state: DialogueState) -> dict:
    """用户在 confirming 阶段说话了。判断是 yes/no/其他。"""
    text = state["user_text"]
    if _YES_PATTERN.match(text):
        return {"next_action": "execute_inquiry"}
    if _NO_PATTERN.match(text):
        return {
            "next_action": "reset_inquiry",
            "reply": "已取消本次下单。如需重新询价请告诉我新的指令。",
            "stage": "init",
            "slots": {},
            "last_question": "",
            "recent_turns": [{"role": "assistant", "text": "已取消本次下单。"}],
        }
    # 用户既没说 yes 也没说 no,可能是补充信息或换话题
    # 把 next_action 设回 classify 再走一遍(简化:走 chat 兜底)
    return {"next_action": "ambiguous_confirm",
            "reply": "请明确回复 'y' 确认或 'n' 取消。",
            "last_question": "请明确回复 'y' 或 'n'",
            "recent_turns": [{"role": "assistant", "text": "请明确回复 'y' 确认或 'n' 取消。"}]}


# ============================================================================
# 6. execute_inquiry_node —— 调下单 API
# ============================================================================

def execute_inquiry_node(state: DialogueState) -> dict:
    """槽位齐全且用户已确认,调 API 下单。"""
    slots = state.get("slots") or {}
    idem_key = f"{state.get('session_id', 'unknown')}:{state.get('turn_count', 0)}"
    result = inquiry_api.submit(slots, idempotency_key=idem_key)

    if result.get("success"):
        reply = (
            f"✓ 下单成功\n"
            f"  订单号: {result['order_id']}\n"
            f"  状态: {result['status_text']}"
        )
    else:
        reply = f"✗ 下单失败: {result.get('error', '未知错误')}"

    return {
        "reply": reply,
        "stage": "executed",
        "slots": {},  # 清空,准备下一笔
        "last_question": "",
        "recent_turns": [{"role": "assistant", "text": reply}],
    }


# ============================================================================
# 7. execute_query_node —— 调查询 API
# ============================================================================

def execute_query_node(state: DialogueState) -> dict:
    """订单/成交查询。结果用模板渲染(关键事实绝不交给 LLM 改写)。"""
    result = query_api.search(filters=state.get("slots") or {})

    if not result.get("success"):
        reply = f"查询失败: {result.get('error', '未知错误')}"
    elif result.get("total", 0) == 0:
        reply = "未查询到符合条件的订单。"
    else:
        lines = [f"共查到 {result['total']} 笔订单:"]
        for o in result["orders"]:
            lines.append(
                f"  • [{o['id']}] {o['bond']} {o['direction'].upper()} "
                f"{o['price']} × {o['amount']} — {o['status']}"
            )
        reply = "\n".join(lines)

    return {
        "reply": reply,
        "stage": "executed",
        "last_question": "",
        "recent_turns": [{"role": "assistant", "text": reply}],
    }


# ============================================================================
# 8. execute_selection_node —— 调智能选债 API
# ============================================================================

def execute_selection_node(state: DialogueState) -> dict:
    """智能选债推荐。"""
    result = selection_api.recommend(criteria=state.get("slots") or {})

    if not result.get("success") or not result.get("bonds"):
        reply = "暂无符合条件的推荐债券。"
    else:
        lines = ["推荐如下:"]
        for b in result["bonds"]:
            lines.append(
                f"  • {b['code']} ({b['name']}) "
                f"收益率 {b['yield']}% / 久期 {b['duration']} / 活跃度 {b['active_score']}"
            )
        lines.append("如需对其中某只询价,请告诉我代码、方向、价格、数量。")
        reply = "\n".join(lines)

    return {
        "reply": reply,
        "stage": "executed",
        "last_question": "",
        "recent_turns": [{"role": "assistant", "text": reply}],
    }


# ============================================================================
# 9. execute_cancel_node —— 调撤单 API (撤单意图也属于 bond_inquiry)
# ============================================================================

def execute_cancel_node(state: DialogueState) -> dict:
    text = state["user_text"].lower()
    cancel_all = "all" in text
    order_id = None  # 真实系统这里可能要从 slots 或上下文解析订单号
    idem_key = f"{state.get('session_id', 'unknown')}:{state.get('turn_count', 0)}"
    result = cancel_api.cancel(order_id=order_id, cancel_all=cancel_all,
                                idempotency_key=idem_key)

    if result.get("success"):
        reply = f"✓ {result.get('msg', '撤单成功')}"
        if "cancelled_count" in result:
            reply += f" (共 {result['cancelled_count']} 笔)"
    else:
        reply = f"✗ 撤单失败: {result.get('error', '未知错误')}"

    return {
        "reply": reply,
        "stage": "executed",
        "slots": {},
        "recent_turns": [{"role": "assistant", "text": reply}],
    }
