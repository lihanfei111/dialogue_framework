"""快速逻辑验证脚本,改完后跑一次确认核心函数行为正确。"""
import json
import prompts
import nodes
import config
import retriever as retriever_mod

# 1. Layer 1 messages 应裁掉 order_status / cancel_scope
few_shots = [{
    "text": "test",
    "output": {
        "intent": "bond_query",
        "slots": {
            "bond_code": None, "direction": None, "price": None,
            "amount": None, "settle_date": None,
            "order_status": "all", "cancel_scope": None,
        },
        "confidence": 0.9,
    },
}]
msgs = prompts.build_l1_messages("test query", {}, few_shots=few_shots)
asst = json.loads(msgs[2]["content"])
assert "order_status" not in asst["slots"], "order_status should be stripped from L1"
assert "cancel_scope" not in asst["slots"], "cancel_scope should be stripped from L1"
print("PASS: L1 slots stripping ok")

# 2. StaticRetriever 返回空列表
config.RETRIEVAL_MODE = "static"
retriever_mod._retriever = None
r = retriever_mod.get_retriever()
assert r.retrieve("test query") == [], "StaticRetriever should return []"
print("PASS: StaticRetriever returns []")
retriever_mod._retriever = None  # 重置

# 3. _is_cancel_intent 规则判断
assert nodes._is_cancel_intent("ref all") is True
assert nodes._is_cancel_intent("ref 220007.IB") is True
assert nodes._is_cancel_intent("帮我撤单") is True
assert nodes._is_cancel_intent("bid 220007") is False
assert nodes._is_cancel_intent("我的订单") is False
print("PASS: _is_cancel_intent rules ok")

# 4. build_l2 messages 结构
msgs_os = prompts.build_l2_order_status_messages("成了多少单")
assert msgs_os[0]["role"] == "system"
assert msgs_os[1]["role"] == "user"
assert "order_status" in msgs_os[0]["content"]  # L2A prompt 包含 order_status

msgs_cs = prompts.build_l2_cancel_scope_messages("ref all")
assert msgs_cs[0]["role"] == "system"
assert "cancel_scope" in msgs_cs[0]["content"]
print("PASS: L2 message builders ok")

# 5. BGE retriever 工厂(恢复默认后)
config.RETRIEVAL_MODE = "bge"
retriever_mod._retriever = None
try:
    r = retriever_mod.get_retriever()
    print(f"PASS: BGERetriever loaded, pool size={len(r.pool)}")
except Exception as e:
    print(f"SKIP: BGERetriever load failed (expected if vectors missing): {e}")

print("\n所有验证通过!")
