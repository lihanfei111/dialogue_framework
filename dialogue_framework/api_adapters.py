"""
外部 API 适配层。

当前是占位实现 —— 由于尚无真实交易 API 可接入,所有方法都返回模拟数据。
接入真实 API 时,**只需替换每个方法的内部实现**,对外接口契约保持不变
(参数和返回 dict 的字段都不变),上层 nodes.py 不需要任何改动。

设计要点:
- 每个 API 一个 adapter 类,职责单一
- 写操作(下单/撤单)支持 idempotency_key(防重)
- 每次调用都记到 LAST_API_CALLS,演示脚本可以读出来展示"系统路由到了哪个 API"
"""

import uuid


# 全局列表:每次 API 调用时往里 append 一条记录。
# 演示脚本(main.py)在每轮跑完后会读取并打印,直观体现"系统路由到了哪个 API"。
LAST_API_CALLS: list = []


def _record_call(api_name: str, method: str, params: dict, result: dict):
    LAST_API_CALLS.append({
        "api": api_name, "method": method,
        "params": params, "result": result,
    })


def consume_calls() -> list:
    """取出并清空本轮的调用记录(由演示脚本调用)。"""
    global LAST_API_CALLS
    out = LAST_API_CALLS
    LAST_API_CALLS = []
    return out


class InquiryAPI:
    """询价/下单 API。"""

    def submit(self, slots: dict, idempotency_key: str = None) -> dict:
        # TODO: 接入真实 API 时替换这里的实现
        result = {
            "success": True,
            "order_id": f"ORD{uuid.uuid4().hex[:8].upper()}",
            "status_text": "已挂单,等待对手方响应",
            "slots_echo": slots,
        }
        _record_call("InquiryAPI", "submit",
                     {"slots": slots, "idempotency_key": idempotency_key}, result)
        return result


class CancelAPI:
    """撤单 API。"""

    def cancel(self, order_id: str = None, cancel_all: bool = False,
               idempotency_key: str = None) -> dict:
        # TODO: 接入真实 API 时替换这里的实现
        if cancel_all:
            result = {"success": True, "cancelled_count": 3, "msg": "已撤销当前所有挂单"}
        else:
            result = {"success": True, "order_id": order_id, "msg": "撤单成功"}
        _record_call("CancelAPI", "cancel",
                     {"order_id": order_id, "cancel_all": cancel_all,
                      "idempotency_key": idempotency_key}, result)
        return result


class QueryAPI:
    """订单/成交查询 API。"""

    def search(self, filters: dict = None) -> dict:
        # TODO: 接入真实 API 时替换这里的实现
        result = {
            "success": True,
            "orders": [
                {"id": "ORD12345", "bond": "220007.IB", "direction": "bid",
                 "price": "1.9003", "amount": "4000万", "status": "已成交"},
                {"id": "ORD12346", "bond": "230012.IB", "direction": "ofr",
                 "price": "2.5500", "amount": "2000万", "status": "挂单中"},
            ],
            "total": 2,
        }
        _record_call("QueryAPI", "search", {"filters": filters}, result)
        return result


class SelectionAPI:
    """智能选债 API。"""

    def recommend(self, criteria: dict = None) -> dict:
        # TODO: 接入真实 API 时替换这里的实现
        result = {
            "success": True,
            "bonds": [
                {"code": "230012.IB", "name": "23国债12", "yield": "2.55",
                 "duration": "3.2", "active_score": 0.92},
                {"code": "240003.IB", "name": "24附息国债03", "yield": "2.48",
                 "duration": "3.0", "active_score": 0.88},
            ],
            "criteria_echo": criteria,
        }
        _record_call("SelectionAPI", "recommend", {"criteria": criteria}, result)
        return result


inquiry_api = InquiryAPI()
cancel_api = CancelAPI()
query_api = QueryAPI()
selection_api = SelectionAPI()
