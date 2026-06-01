"""
DialogueState —— 整个对话流程的"状态容器"。
LangGraph 的 checkpointer 会自动把这个对象按 thread_id 存到 SQLite,
每轮 graph.invoke 时自动加载,跑完 node 自动写回。
"""

from typing import TypedDict, Optional, Annotated
import operator


class DialogueState(TypedDict, total=False):
    # ============ 本轮输入 ============
    user_text: str                          # 用户本轮说的话

    # ============ 意图识别结果 ============
    intent: Optional[str]                   # bond_inquiry | bond_query | bond_selection | chat
    intent_confidence: Optional[float]

    # ============ 槽位(跨轮累积填充) ============
    slots: dict                             # 询价: {"bond_code","direction","price",
                                            #        "amount","settle_date"}
                                            # 撤单: {"cancel_scope"}  全撤/指定撤
                                            #        值: "all" | "specified"
                                            # 查询: {"order_status"}  细分订单状态
                                            #        值: "all" | "pending" | "completed"

    # ============ 流程阶段 ============
    # init | collecting_slots | confirming | executed | chatting
    stage: str

    # ============ 本轮输出 ============
    reply: str                              # 给用户的回复(graph 跑完后从这里取)
    last_question: str                      # 系统刚刚问的问题(下一轮要给模型看)
    next_action: str                        # 路由决策结果,内部用

    # ============ 上下文窗口 ============
    # 用 operator.add 作 reducer,这样 node 里返回 {"recent_turns": [new_item]}
    # 会自动追加到现有列表后面,而不是覆盖。
    recent_turns: Annotated[list, operator.add]

    # ============ 元信息 ============
    session_id: str
    turn_count: int


def new_state(session_id: str) -> dict:
    """初始化一个全新会话的 state(给 graph.invoke 用)。"""
    return {
        "session_id": session_id,
        "turn_count": 0,
        "stage": "init",
        "slots": {},
        "recent_turns": [],
        "intent": None,
        "reply": "",
        "last_question": "",
        "next_action": "",
    }


# bond_inquiry 槽位完整性定义 —— 哪些字段缺一不可
INQUIRY_REQUIRED_SLOTS = ["bond_code", "direction", "price", "amount"]
INQUIRY_OPTIONAL_SLOTS = ["settle_date"]


def missing_inquiry_slots(slots: dict) -> list:
    """返回询价场景下还缺的必填槽位。"""
    return [k for k in INQUIRY_REQUIRED_SLOTS if not slots.get(k)]


def all_inquiry_slots_filled(slots: dict) -> bool:
    return len(missing_inquiry_slots(slots)) == 0
