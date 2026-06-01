"""
组装 LangGraph 图 + Redis 状态持久化。

设计选择(为什么不用 LangGraph 官方的 RedisSaver):
  - 官方 RedisSaver 依赖 RediSearch 模块(只在 Redis Stack 自带),普通 Redis 跑不了
  - 我们这里只需要"按 session_id 存取 state 字典"这种简单需求,普通 Redis 命令就够
  - 所以 compile graph 时不传 checkpointer,改成在 handle_turn 里自己读写 Redis

LangGraph 在 graph 内部的状态合并(reducer,如 recent_turns 的 append)依然正常工作 ——
我们只是把"跨 invoke 的持久化"换成了自己管,不影响 graph 内的语义。
"""

import json

import redis
from langgraph.graph import StateGraph, START, END

import config
from state import DialogueState
import nodes
import routes


# ============ 单例:graph + redis client ============

_graph = None
_redis = None


def _get_redis() -> redis.Redis:
    """单例 Redis 连接。decode_responses=True 让 get 直接返回 str 而不是 bytes。"""
    global _redis
    if _redis is None:
        _redis = redis.Redis.from_url(config.REDIS_URL, decode_responses=True)
    return _redis


def build_graph():
    """构建并编译对话流程图。注意:不绑定 checkpointer,跨轮持久化由 handle_turn 自己做。"""
    builder = StateGraph(DialogueState)

    # ============ 注册节点 ============
    builder.add_node("classify", nodes.classify_node)
    builder.add_node("chat_reply", nodes.chat_reply_node)
    builder.add_node("ask_missing_slots", nodes.ask_missing_slots_node)
    builder.add_node("ask_confirm", nodes.ask_confirm_node)
    builder.add_node("handle_confirm", nodes.handle_confirm_node)
    builder.add_node("execute_inquiry", nodes.execute_inquiry_node)
    builder.add_node("execute_query", nodes.execute_query_node)
    builder.add_node("execute_selection", nodes.execute_selection_node)
    builder.add_node("execute_cancel", nodes.execute_cancel_node)

    # ============ 起点和路由 ============
    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        routes.route_after_classify,
        {
            "chat_reply": "chat_reply",
            "ask_missing_slots": "ask_missing_slots",
            "ask_confirm": "ask_confirm",
            "handle_confirm": "handle_confirm",
            "execute_query": "execute_query",
            "execute_selection": "execute_selection",
            "execute_cancel": "execute_cancel",
        },
    )
    builder.add_conditional_edges(
        "handle_confirm",
        routes.route_after_confirm,
        {"execute_inquiry": "execute_inquiry", "__end__": END},
    )
    builder.add_edge("chat_reply", END)
    builder.add_edge("ask_missing_slots", END)
    builder.add_edge("ask_confirm", END)
    builder.add_edge("execute_inquiry", END)
    builder.add_edge("execute_query", END)
    builder.add_edge("execute_selection", END)
    builder.add_edge("execute_cancel", END)

    # 注意:这里 compile 不传 checkpointer
    return builder.compile()


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ============ Redis 读写辅助 ============

def _session_key(session_id: str) -> str:
    return f"{config.SESSION_KEY_PREFIX}{session_id}"


def load_session(session_id: str) -> dict:
    """从 Redis 读出 session state。session 不存在则返回空 dict。"""
    raw = _get_redis().get(_session_key(session_id))
    if raw is None:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # 反序列化失败按新会话处理,避免脏数据卡死整个 session
        return {}


def save_session(session_id: str, state: dict):
    """把 state 写回 Redis,带 TTL。"""
    r = _get_redis()
    key = _session_key(session_id)
    payload = json.dumps(state, ensure_ascii=False, default=str)
    if config.SESSION_TTL_SECONDS:
        r.setex(key, config.SESSION_TTL_SECONDS, payload)
    else:
        r.set(key, payload)


def delete_session(session_id: str):
    """显式删除一个 session(比如用户主动结束会话时)。"""
    _get_redis().delete(_session_key(session_id))


def _trim_recent_turns(state: dict, max_turns: int):
    """对 recent_turns 做截断,防止 state 在 Redis 里无限膨胀。"""
    rt = state.get("recent_turns")
    if isinstance(rt, list) and len(rt) > max_turns * 2:  # *2 因为一轮含 user+assistant
        state["recent_turns"] = rt[-max_turns * 2:]


# ============ 主入口 ============

def handle_turn(session_id: str, user_text: str) -> dict:
    """
    一轮对话的入口。每来一条用户消息就调一次。

    流程(对应教程"每轮三件事:更新状态、决定下一动作、生成输出"):
      1. 从 Redis 读出该 session 的历史 state
      2. 合并本轮 user_text 进去
      3. 跑 LangGraph(graph 内部 reducer 仍会正常工作)
      4. 截断超长的 recent_turns
      5. 把新 state 写回 Redis
    """
    graph = get_graph()

    # 1. 读历史 state
    state = load_session(session_id)

    # 2. 本轮新输入
    state["user_text"] = user_text
    state["session_id"] = session_id

    # 3. 跑图
    final_state = graph.invoke(state)

    # 4. 截断 recent_turns
    _trim_recent_turns(final_state, config.MAX_RECENT_TURNS)

    # 5. 写回 Redis
    save_session(session_id, final_state)

    return final_state
