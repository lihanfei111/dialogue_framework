"""
LLM 客户端。统一打真实 endpoint(OpenAI 兼容协议),自带重试和延迟统计。
"""

import time

from openai import OpenAI, APIError, APITimeoutError, APIConnectionError

import config


_client = None


def _get_client() -> OpenAI:
    """单例 client,避免每次新建连接。"""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=config.API_KEY,
            base_url=config.BASE_URL,
            timeout=config.TIMEOUT,
        )
    return _client


def chat(messages: list,
         temperature: float = None,
         max_tokens: int = None) -> dict:
    """
    调一次大模型。返回 dict:
      {
        "content": 模型输出文本(去前后空白),
        "latency_ms": 调用耗时(毫秒, int),
        "error": None 或 错误字符串,
      }
    内部已带重试。所有异常都被吞掉返回 error 字段,调用方不需要 try。
    """
    client = _get_client()
    temperature = config.TEMPERATURE if temperature is None else temperature
    max_tokens = config.MAX_TOKENS if max_tokens is None else max_tokens

    last_err = None
    t0 = time.perf_counter()
    for attempt in range(config.MAX_RETRIES + 1):
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=config.MODEL_ID,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"thinking": {"type": "disabled"}, "enable_thinking": False},
            )
            usage = resp.usage
            msg = resp.choices[0].message
            content = (msg.content or "").strip()
            # 兜底:部分实现把答案放 reasoning_content、content 留空
            if not content:
                content = (getattr(msg, "reasoning_content", None) or "").strip()
            if config.DEBUG_LLM:
                rc = getattr(msg, "reasoning_content", None) or ""
                print(
                    f"\n[DEBUG] content({len(content)}): {content[:120]!r}"
                    f"\n[DEBUG] reasoning_content({len(rc)}): {rc[:80]!r}",
                    flush=True,
                )
            return {
                "content": content,
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "error": None,
                "retries":           attempt,
                "prompt_tokens":     int(getattr(usage, "prompt_tokens",     0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens":      int(getattr(usage, "total_tokens",      0) or 0),
            }
        except (APITimeoutError, APIConnectionError) as e:
            # 网络/超时类错误重试
            last_err = f"{type(e).__name__}: {e}"
            if attempt < config.MAX_RETRIES:
                time.sleep(0.5 * (attempt + 1))
                continue
        except APIError as e:
            last_err = f"APIError: {e}"
            break
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            break

    return {
        "content": "",
        "latency_ms": int((time.perf_counter() - t0) * 1000),
        "error": last_err,
        "retries":           config.MAX_RETRIES,
        "prompt_tokens":     0,
        "completion_tokens": 0,
        "total_tokens":      0,
    }


# ============================================================================
# 多温度投票调用(USE_VOTING=True 时使用)
# ============================================================================

def _try_parse_json(text):
    """宽松 JSON 解析,用于投票内部合并。"""
    import json as _json
    import re
    text = (text or "").strip()
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()
    try:
        return _json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return _json.loads(m.group(0))
            except Exception:
                return None
        return None


def _merge_voted_results(results: list) -> dict:
    """对每个字段独立投票。results 是 [{"parsed":..., "confidence":...}, ...]"""
    import json as _json
    from collections import Counter

    def vote_field(values_with_conf):
        """多数胜出;tie 时取 confidence 最高;全 None 返回 None。"""
        non_null = [(v, c) for v, c in values_with_conf if v is not None]
        if not non_null:
            return None
        counter = Counter(v for v, _ in non_null)
        max_count = max(counter.values())
        candidates = [v for v, c in counter.items() if c == max_count]
        if len(candidates) == 1:
            return candidates[0]
        best_val, best_conf = None, -1.0
        for v, c in non_null:
            if v in candidates and c > best_conf:
                best_val, best_conf = v, c
        return best_val

    intent = vote_field([(r["parsed"].get("intent"), r["confidence"]) for r in results])

    all_slot_keys = set()
    for r in results:
        all_slot_keys.update((r["parsed"].get("slots") or {}).keys())

    slots = {}
    for key in all_slot_keys:
        vals = [((r["parsed"].get("slots") or {}).get(key), r["confidence"]) for r in results]
        # 槽位保守规则:至少 2 次填了相同非空值才取,否则 null
        non_null = [(v, c) for v, c in vals if v is not None and v != ""]
        if len(non_null) < 2:
            slots[key] = None
        else:
            slots[key] = vote_field(vals)

    confidence = sum(r["confidence"] for r in results) / len(results)
    return {"intent": intent, "slots": slots, "confidence": round(confidence, 3)}


def vote_chat(messages: list) -> dict:
    """多温度投票:同一 messages 用 3 个不同 temperature 各调一次,合并结果。
    返回格式与 chat() 一致(content 是合并后的 JSON 字符串)。"""
    import json as _json
    import config as _config

    results = []
    total_latency = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_total_tokens = 0
    last_error = None

    for temp in _config.VOTING_TEMPERATURES:
        r = chat(messages, temperature=temp)
        total_latency += (r.get("latency_ms") or 0)
        total_prompt_tokens += r.get("prompt_tokens", 0)
        total_completion_tokens += r.get("completion_tokens", 0)
        total_total_tokens += r.get("total_tokens", 0)
        if r.get("error"):
            last_error = r["error"]
            continue
        parsed = _try_parse_json(r["content"])
        if parsed is not None:
            results.append({"parsed": parsed, "confidence": parsed.get("confidence", 0.0)})

    if not results:
        return {
            "content": "",
            "latency_ms": total_latency,
            "error": last_error or "all_voting_calls_failed",
            "retries": 0,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_total_tokens,
        }

    merged = _merge_voted_results(results)
    return {
        "content": _json.dumps(merged, ensure_ascii=False),
        "latency_ms": total_latency,
        "error": None,
        "retries": 0,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_total_tokens,
    }
