"""
路由函数。每个函数返回下一个节点的名字(字符串)。
LangGraph 调 add_conditional_edges 时会用这些函数决定路由方向。
"""

import re
from state import DialogueState, all_inquiry_slots_filled


def route_after_classify(state: DialogueState) -> str:
    """
    classify 之后的主路由。根据 intent + 当前 stage 决定下一个 node。

    特殊处理:如果上轮 stage 已经是 confirming(等待用户确认),
    本轮无论模型识别出什么意图,都先走 handle_confirm 节点判断是 yes/no/其他。
    """
    stage = state.get("stage", "init")
    intent = state.get("intent", "chat")

    # 优先级 1:用户在确认阶段的回复
    if stage == "confirming":
        return "handle_confirm"

    # 优先级 2:按 intent 路由
    if intent == "chat":
        return "chat_reply"

    if intent == "bond_inquiry":
        # 是不是撤单?
        text = state.get("user_text", "").lower()
        if re.search(r"\bref\b|撤单", text):
            return "execute_cancel"
        # 询价:看槽位
        slots = state.get("slots") or {}
        if all_inquiry_slots_filled(slots):
            return "ask_confirm"
        return "ask_missing_slots"

    if intent == "bond_query":
        return "execute_query"

    if intent == "bond_selection":
        return "execute_selection"

    # 兜底
    return "chat_reply"


def route_after_confirm(state: DialogueState) -> str:
    """
    handle_confirm 之后的路由。
    用户说了 yes → execute_inquiry
    用户说了 no → END(reset 已在 handle_confirm 里做了)
    用户含糊 → END(已在 handle_confirm 里回了"请明确")
    """
    next_action = state.get("next_action", "")
    if next_action == "execute_inquiry":
        return "execute_inquiry"
    # reset 和 ambiguous 都直接结束本轮
    return "__end__"
