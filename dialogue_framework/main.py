"""
对话演示入口 —— 给领导/同事看效果用。

用法:
    python main.py                  # 跑内置的 6 个演示场景
    python main.py --interactive    # 进入交互模式,手动输入对话
    python main.py --scenario 1     # 只跑第 1 个场景

跑前提示:
    需要 VPN 通,能访问 config.py 里配置的 LLM endpoint。
"""

import argparse
import json
import sys

import config
from api_adapters import consume_calls
from graph import handle_turn


# ============ 输出样式 ============

USE_COLOR = sys.stdout.isatty()


def _c(text, code):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def header(s):
    return _c(s, "1;36")           # 加粗青色

def user_label():
    return _c("[用户]", "1;33")    # 加粗黄色

def bot_label():
    return _c("[系统]", "1;32")    # 加粗绿色

def meta(s):
    return _c(s, "90")             # 暗灰

def api(s):
    return _c(s, "1;35")           # 加粗紫色


# ============ 单轮的漂亮渲染 ============

STAGE_CN = {
    "init": "初始", "collecting_slots": "收集槽位中",
    "confirming": "等待确认", "executed": "已执行",
    "chatting": "闲聊中",
}


def render_turn(turn_no: int, user_text: str, state: dict):
    """把一轮对话漂亮地打出来。"""
    print(f"\n{user_label()} {user_text}")

    # 系统回复(reply 可能多行,缩进对齐)
    reply = state.get("reply", "(无回复)")
    reply_lines = reply.split("\n")
    print(f"{bot_label()} {reply_lines[0]}")
    for line in reply_lines[1:]:
        print(f"        {line}")

    # 内部状态(给技术同事看)
    intent = state.get("intent")
    stage = state.get("stage")
    slots = state.get("slots") or {}
    slots_str = ", ".join(f"{k}={v}" for k, v in slots.items() if v) or "(空)"
    print(meta(f"   ├─ 意图: {intent}"))
    print(meta(f"   ├─ 阶段: {stage} ({STAGE_CN.get(stage, '?')})"))
    print(meta(f"   └─ 槽位: {slots_str}"))

    # 本轮如果触发了 API 调用,把调用信息显式打出来
    calls = consume_calls()
    for c in calls:
        params_str = ", ".join(f"{k}={v}" for k, v in c["params"].items() if v is not None)
        print(api(f"   🔧 [API 调用] {c['api']}.{c['method']}({params_str})"))
        # 简化输出 result
        result = c["result"]
        key_info = {k: v for k, v in result.items() if k != "slots_echo" and k != "criteria_echo"}
        print(api(f"      返回: {json.dumps(key_info, ensure_ascii=False)}"))


# ============ 场景定义 ============

SCENARIOS = [
    {
        "name": "场景 1:多轮追问与确认 —— 询价主流程",
        "session_id": "demo_1",
        "turns": [
            "Bid 220007.IB 4000万",     # 缺 price,系统应追问
            "1.9003 今+1",              # 补齐 → 系统核对
            "y",                        # 确认 → 系统下单
        ],
    },
    {
        "name": "场景 2:查询订单",
        "session_id": "demo_2",
        "turns": ["我的订单"],
    },
    {
        "name": "场景 3:智能选债",
        "session_id": "demo_3",
        "turns": ["帮我推荐3年期国债"],
    },
    {
        "name": "场景 4:撤单",
        "session_id": "demo_4",
        "turns": ["ref all"],
    },
    {
        "name": "场景 5:用户取消确认 —— 风险控制演示",
        "session_id": "demo_5",
        "turns": [
            "Bid 220007.IB 4000万 1.9003",
            "n",                        # 用户回 n,系统应取消
        ],
    },
    {
        "name": "场景 6:先闲聊后询价 —— 上下文延续能力",
        "session_id": "demo_6",
        "turns": [
            "bid 是什么意思",            # chat:解释术语
            "好的,Bid 220007.IB 4000万 1.9003 今+1",  # 完整询价
            "y",
        ],
    },
]


def run_scenario(scenario: dict):
    print("\n" + header("═" * 70))
    print(header(scenario["name"]))
    print(header(f"session_id: {scenario['session_id']}"))
    print(header("═" * 70))

    for i, text in enumerate(scenario["turns"], 1):
        state = handle_turn(scenario["session_id"], text)
        render_turn(i, text, state)


def interactive():
    print(header("=" * 70))
    print(header("交互模式 —— 输入 'exit' 退出,'reset' 重开会话"))
    print(meta(f"模型: {config.MODEL_ID}"))
    print(header("=" * 70))

    session_id = "interactive_001"
    turn_no = 0
    counter = 0
    while True:
        try:
            text = input(f"\n{user_label()} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text == "exit":
            break
        if text == "reset":
            counter += 1
            session_id = f"interactive_{counter+1:03d}"
            turn_no = 0
            print(meta(f"(已切换新会话 {session_id})"))
            continue
        if not text:
            continue
        turn_no += 1
        state = handle_turn(session_id, text)
        # interactive 模式下不重复打印用户输入(已经显示在 prompt 里)
        reply = state.get("reply", "")
        for i, line in enumerate(reply.split("\n")):
            print(f"{bot_label() if i == 0 else '        '} {line}")
        intent = state.get("intent")
        stage = state.get("stage")
        slots = state.get("slots") or {}
        slots_str = ", ".join(f"{k}={v}" for k, v in slots.items() if v) or "(空)"
        print(meta(f"   └─ intent={intent} | stage={stage} | slots={slots_str}"))
        for c in consume_calls():
            params_str = ", ".join(f"{k}={v}" for k, v in c["params"].items() if v is not None)
            print(api(f"   🔧 [API] {c['api']}.{c['method']}({params_str})"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="进入交互式 REPL")
    parser.add_argument("--scenario", "-s", type=int, default=0,
                        help="只跑指定编号的场景(1-6),0=全部")
    args = parser.parse_args()

    if args.interactive:
        interactive()
        return

    print(header("\n" + "═" * 70))
    print(header("FICC 债券机器人 —— 多轮对话演示"))
    print(meta(f"  模型: {config.MODEL_ID}"))
    print(meta(f"  注:交易 API 当前是占位实现,接入真实系统时替换 api_adapters.py 即可"))
    print(header("═" * 70))

    if args.scenario > 0:
        if 1 <= args.scenario <= len(SCENARIOS):
            run_scenario(SCENARIOS[args.scenario - 1])
        else:
            print(f"无效场景编号: {args.scenario}")
    else:
        for sc in SCENARIOS:
            run_scenario(sc)

    print("\n" + header("═" * 70))
    print(header("演示结束"))
    print(header("═" * 70))


if __name__ == "__main__":
    main()
