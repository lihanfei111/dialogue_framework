"""
意图识别评测脚本。

复用 dialogue_framework 内部的 build_classify_messages 和 llm_client,
做单轮(无对话上下文)意图分类的批量评测。

输出格式跟 intent_framework/eval.py 完全一致:
  outputs/<tag>_results.csv     全量预测明细
  outputs/<tag>_bad_cases.csv   错误 case
  outputs/<tag>_report.txt      准确率/F1/混淆矩阵/延迟

用法:
    python eval_intent.py                    # 全量评测
    python eval_intent.py --limit 20         # 只跑前 20 条
    python eval_intent.py --dry-run          # 不调 LLM,只验证数据加载
    python eval_intent.py --output-tag v2    # 给输出文件加标签
"""
from __future__ import annotations   # 让 PEP 604/585 注解在 Python 3.9 也能用

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import config
from llm_client import chat
from prompts import build_classify_messages


# ============================================================================
# 测试集加载:从 test_data.jsonl 读(含 excel 原文 + 发散生成数据)
# ============================================================================

def load_test_data(path: str = None) -> list:
    """读 test_data.jsonl。返回 samples,字段对齐既有评测代码:
       label = intent, gold_order_status, source_sheet = source。"""
    path = path or config.TEST_DATA_PATH
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            samples.append({
                "text": e["text"],
                "label": e["intent"],
                "gold_order_status": e.get("order_status"),
                "gold_cancel_scope": e.get("cancel_scope"),
                "source_sheet": e.get("source", ""),
            })
    return samples


# ============================================================================
# 单条评测:复用 dialogue_framework 的 prompt,但传空 state(单轮)
# ============================================================================

_JSON_PATTERN = re.compile(r"\{[\s\S]*\}")
_SLOT_KEYS = ("bond_code", "direction", "price", "amount",
              "settle_date", "order_status", "cancel_scope")


def _parse_json(text: str) -> dict | None:
    """多级兜底解析：thinking 剥离 → 直接解析 → 正则提取 → 逐步修复 → 字段级提取。"""
    if not text:
        return None
    cleaned = text.strip()

    # 1. 剥离 thinking 块
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned).strip()

    # 2. 剥离 markdown 代码块
    if "```" in cleaned:
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    # 3. 直接解析
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 4. 提取 {...} 候选块，逐步修复后重试
    m = _JSON_PATTERN.search(cleaned)
    candidate = m.group(0) if m else cleaned
    for attempt in (
        candidate,
        re.sub(r",\s*([}\]])", r"\1", candidate),                          # 尾随逗号
        candidate.replace("'", '"'),                                        # 单引号→双引号
        candidate + "}" * max(0, candidate.count("{") - candidate.count("}")),  # 截断补齐
    ):
        try:
            return json.loads(attempt)
        except Exception:
            continue

    # 5. 字段级兜底：至少提取 intent
    m_intent = re.search(r'"intent"\s*:\s*"([^"]+)"', cleaned)
    if m_intent:
        return {"intent": m_intent.group(1), "slots": {}, "confidence": 0.5}

    return None


def _normalize_output(parsed: dict) -> dict:
    """确保解析结果字段完整、类型正确，缺失字段补默认值。"""
    slots = dict(parsed.get("slots") or {})
    for k in _SLOT_KEYS:
        slots.setdefault(k, None)
    try:
        conf = float(parsed.get("confidence") or 0.5)
        conf = max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        conf = 0.5
    return {"intent": parsed.get("intent") or "chat", "slots": slots, "confidence": conf}


def classify_single(text: str) -> dict:
    """对一条文本做单轮意图识别。state 传 {},走"无上下文"分支(等价单轮分类)。
    若启用动态 few-shot,则按 text 召回最相关示例注入 prompt。"""
    few_shots = None
    if config.USE_DYNAMIC_FEWSHOT and config.RETRIEVAL_MODE != "static":
        from retriever import get_retriever
        few_shots = get_retriever().retrieve(text)
    messages = build_classify_messages(text, {}, few_shots=few_shots)
    if config.USE_VOTING:
        from llm_client import vote_chat
        result = vote_chat(messages)
    else:
        result = chat(messages)

    out = {
        "text": text,
        "intent": None,
        "slots": {},
        "order_status": None,
        "cancel_scope": None,
        "confidence": None,
        "latency_ms":        result["latency_ms"],
        "error":             result["error"],
        "raw_output":        result["content"],
        "retries":           result.get("retries", 0),
        "prompt_tokens":     result.get("prompt_tokens", 0),
        "completion_tokens": result.get("completion_tokens", 0),
        "total_tokens":      result.get("total_tokens", 0),
    }
    if result["error"]:
        return out

    parsed = _parse_json(result["content"])
    if parsed is None:
        out["error"] = "json_parse_failed"
        return out

    parsed = _normalize_output(parsed)
    intent = parsed.get("intent")
    if intent not in config.INTENT_LABELS:
        out["error"] = "invalid_intent_label"
    out["intent"] = intent
    out["slots"] = parsed.get("slots") or {}
    out["order_status"] = (out["slots"] or {}).get("order_status")
    out["cancel_scope"] = (out["slots"] or {}).get("cancel_scope")
    out["confidence"] = parsed.get("confidence")
    return out


# ============================================================================
# 评测主流程(并发、指标计算、报告生成 —— 跟 intent_framework 一致)
# ============================================================================

def _print_header(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


def run_predictions(samples, concurrency):
    results = [None] * len(samples)
    done = 0
    total = len(samples)
    t_start = time.perf_counter()

    def _work(idx):
        return idx, classify_single(samples[idx]["text"])

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(_work, i) for i in range(total)]
        for fut in as_completed(futures):
            idx, pred = fut.result()
            results[idx] = pred
            done += 1

            latency = pred.get("latency_ms") or 0
            retries = pred.get("retries") or 0
            error   = pred.get("error")
            suffix  = ""
            if error:
                suffix = f"  ✗ {error}"
            elif retries:
                suffix = f"  ⚠ 重试{retries}次"
            print(f"  [{done:3d}/{total}] #{idx+1:3d}  {latency:6d}ms{suffix}", flush=True)
            if error == "json_parse_failed":
                raw = (pred.get("raw_output") or "").strip()
                preview = raw[:300] + ("..." if len(raw) > 300 else "")
                print(f"    ↳ 原始输出: {preview}", flush=True)

            if done % 10 == 0 or done == total:
                elapsed = time.perf_counter() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  --- 进度 {done}/{total}  用时 {elapsed:.1f}s  ETA {eta:.1f}s ---", flush=True)
    return results


def compute_metrics(samples, preds):
    n = len(samples)
    correct = sum(1 for s, p in zip(samples, preds) if p["intent"] == s["label"])
    parse_errors = sum(1 for p in preds if p["error"])
    api_errors = sum(1 for p in preds
                     if p["error"] and p["error"] not in ("json_parse_failed", "invalid_intent_label"))

    labels = sorted(set(s["label"] for s in samples)
                    | set(p["intent"] for p in preds if p["intent"]))
    per_class = {}
    for L in labels:
        tp = sum(1 for s, p in zip(samples, preds) if s["label"] == L and p["intent"] == L)
        fp = sum(1 for s, p in zip(samples, preds) if s["label"] != L and p["intent"] == L)
        fn = sum(1 for s, p in zip(samples, preds) if s["label"] == L and p["intent"] != L)
        support = sum(1 for s in samples if s["label"] == L)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[L] = {"precision": precision, "recall": recall, "f1": f1,
                        "support": support, "tp": tp, "fp": fp, "fn": fn}

    matrix = {gl: Counter() for gl in labels}
    for s, p in zip(samples, preds):
        pred_label = p["intent"] if p["intent"] else "<ERROR>"
        matrix[s["label"]][pred_label] += 1
    all_pred_labels = sorted({lbl for c in matrix.values() for lbl in c})

    latencies = [p["latency_ms"] for p in preds if p["latency_ms"] is not None]
    latencies_sorted = sorted(latencies)
    def pct(p):
        if not latencies_sorted:
            return 0
        return latencies_sorted[min(int(len(latencies_sorted) * p), len(latencies_sorted) - 1)]

    total_retries  = sum(p.get("retries", 0) for p in preds)
    total_failures = sum(1 for p in preds if p["error"])
    pt = [p.get("prompt_tokens", 0)     for p in preds]
    ct = [p.get("completion_tokens", 0) for p in preds]
    tt = [p.get("total_tokens", 0)      for p in preds]

    # ===== order_status 细分评测(仅 gold=bond_query 的样本) =====
    os_total = 0
    os_correct = 0
    os_confusion = {s: Counter() for s in config.ORDER_STATUS_LABELS}
    for s, p in zip(samples, preds):
        if s["label"] != "bond_query":
            continue
        gold_os = s.get("gold_order_status") or "all"
        os_total += 1
        if p["intent"] == "bond_query":
            pred_os = p.get("order_status") or "all"
            if pred_os not in config.ORDER_STATUS_LABELS:
                pred_os = "<INVALID>"
        else:
            pred_os = "<意图判错>"
        if pred_os == gold_os:
            os_correct += 1
        if gold_os in os_confusion:
            os_confusion[gold_os][pred_os] += 1

    os_all_pred = sorted({lbl for c in os_confusion.values() for lbl in c})

    # ===== cancel_scope 细分评测(仅 gold=bond_inquiry 且是撤单的样本) =====
    # 撤单样本 = gold_cancel_scope 不为 None
    cs_total = 0
    cs_correct = 0
    cs_confusion = {s: Counter() for s in config.CANCEL_SCOPE_LABELS}
    for s, p in zip(samples, preds):
        gold_cs = s.get("gold_cancel_scope")
        if gold_cs is None:
            continue
        cs_total += 1
        if p["intent"] == "bond_inquiry":
            pred_cs = p.get("cancel_scope")
            if pred_cs is None:
                pred_cs = "<未填>"
            elif pred_cs not in config.CANCEL_SCOPE_LABELS:
                pred_cs = "<INVALID>"
        else:
            pred_cs = "<意图判错>"
        if pred_cs == gold_cs:
            cs_correct += 1
        if gold_cs in cs_confusion:
            cs_confusion[gold_cs][pred_cs] += 1

    cs_all_pred = sorted({lbl for c in cs_confusion.values() for lbl in c})

    return {
        "total": n, "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "parse_or_label_errors": parse_errors, "api_errors": api_errors,
        "labels": labels, "per_class": per_class,
        "confusion": matrix, "all_pred_labels": all_pred_labels,
        "order_status": {
            "total": os_total, "correct": os_correct,
            "accuracy": os_correct / os_total if os_total else 0.0,
            "confusion": os_confusion, "all_pred": os_all_pred,
        },
        "cancel_scope": {
            "total": cs_total, "correct": cs_correct,
            "accuracy": cs_correct / cs_total if cs_total else 0.0,
            "confusion": cs_confusion, "all_pred": cs_all_pred,
        },
        "latency_ms": {
            "mean": sum(latencies) / len(latencies) if latencies else 0,
            "p50": pct(0.50), "p90": pct(0.90), "p99": pct(0.99),
            "max": max(latencies) if latencies else 0,
        },
        "call_stats": {
            "requests": n,
            "retries":  total_retries,
            "failures": total_failures,
        },
        "token_usage": {
            "prompt_total":     sum(pt), "prompt_mean":     sum(pt) / len(pt) if pt else 0, "prompt_max":     max(pt) if pt else 0,
            "completion_total": sum(ct), "completion_mean": sum(ct) / len(ct) if ct else 0, "completion_max": max(ct) if ct else 0,
            "total_total":      sum(tt), "total_mean":      sum(tt) / len(tt) if tt else 0, "total_max":      max(tt) if tt else 0,
        },
    }


def render_report(metrics):
    lines = []
    lines.append("意图识别评测报告 (dialogue_framework / 单轮模式)")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"模型: {config.MODEL_ID}")
    lines.append("")
    lines.append(f"样本总数: {metrics['total']}")
    lines.append(f"预测正确: {metrics['correct']}")
    lines.append(f"准确率  : {metrics['accuracy']:.2%}")
    lines.append(f"解析/标签异常: {metrics['parse_or_label_errors']}  (其中 API 错误: {metrics['api_errors']})")
    lines.append("")
    lines.append("--- 分类指标 (per class) ---")
    lines.append(f"{'label':<18s}{'precision':>10s}{'recall':>10s}{'f1':>8s}{'support':>10s}")
    for L, m in metrics["per_class"].items():
        lines.append(f"{L:<18s}{m['precision']:>10.3f}{m['recall']:>10.3f}{m['f1']:>8.3f}{m['support']:>10d}")
    lines.append("")
    lines.append("--- 混淆矩阵 (行=真实, 列=预测) ---")
    headers = metrics["all_pred_labels"]
    col_w = max(8, max(len(h) for h in headers) + 1) if headers else 8
    lines.append(" " * 16 + "".join(f"{h:>{col_w}s}" for h in headers))
    for gl in metrics["labels"]:
        row = metrics["confusion"][gl]
        lines.append(f"{gl:<16s}" + "".join(f"{row.get(h, 0):>{col_w}d}" for h in headers))
    lines.append("")
    lat = metrics["latency_ms"]
    lines.append("--- 调用延迟 (ms) ---")
    lines.append(f"  mean = {lat['mean']:.0f}   p50 = {lat['p50']}   p90 = {lat['p90']}   p99 = {lat['p99']}   max = {lat['max']}")
    lines.append("")

    cs = metrics.get("call_stats", {})
    retry_rate = cs["retries"] / cs["requests"] * 100 if cs.get("requests") else 0
    lines.append("--- 调用统计 ---")
    lines.append(f"  总请求数: {cs.get('requests', 0)}   重试次数: {cs.get('retries', 0)} (重试率 {retry_rate:.1f}%)   失败次数: {cs.get('failures', 0)}")
    lines.append("")

    tu = metrics.get("token_usage", {})
    lines.append("--- Token 消耗 ---")
    lines.append(f"  {'':12s}{'总计':>10s}{'均值':>10s}{'最大':>10s}")
    lines.append(f"  {'prompt':<12s}{tu.get('prompt_total', 0):>10d}{tu.get('prompt_mean', 0):>10.0f}{tu.get('prompt_max', 0):>10d}")
    lines.append(f"  {'completion':<12s}{tu.get('completion_total', 0):>10d}{tu.get('completion_mean', 0):>10.0f}{tu.get('completion_max', 0):>10d}")
    lines.append(f"  {'total':<12s}{tu.get('total_total', 0):>10d}{tu.get('total_mean', 0):>10.0f}{tu.get('total_max', 0):>10d}")
    lines.append("")

    # ===== order_status 细分报告 =====
    os_m = metrics.get("order_status", {})
    if os_m.get("total"):
        lines.append("--- bond_query 细分订单状态 (order_status) ---")
        lines.append(f"  评测样本(gold=bond_query): {os_m['total']}")
        lines.append(f"  状态判对: {os_m['correct']}")
        lines.append(f"  状态准确率: {os_m['accuracy']:.2%}")
        lines.append("  混淆矩阵 (行=真实状态, 列=预测):")
        headers = os_m["all_pred"]
        col_w = max(10, max((len(h) for h in headers), default=8) + 1)
        lines.append(" " * 12 + "".join(f"{h:>{col_w}s}" for h in headers))
        for gs in config.ORDER_STATUS_LABELS:
            row = os_m["confusion"].get(gs, {})
            if sum(row.values()) == 0:
                continue
            lines.append(f"{gs:<12s}" + "".join(f"{row.get(h, 0):>{col_w}d}" for h in headers))
        lines.append("")

    # ===== cancel_scope 细分报告 =====
    cs_m = metrics.get("cancel_scope", {})
    if cs_m.get("total"):
        lines.append("--- bond_inquiry 撤单细分 (cancel_scope) ---")
        lines.append(f"  评测样本(撤单 gold): {cs_m['total']}")
        lines.append(f"  细分判对: {cs_m['correct']}")
        lines.append(f"  细分准确率: {cs_m['accuracy']:.2%}")
        lines.append("  混淆矩阵 (行=真实, 列=预测):")
        headers = cs_m["all_pred"]
        col_w = max(10, max((len(h) for h in headers), default=8) + 1)
        lines.append(" " * 12 + "".join(f"{h:>{col_w}s}" for h in headers))
        for gs in config.CANCEL_SCOPE_LABELS:
            row = cs_m["confusion"].get(gs, {})
            if sum(row.values()) == 0:
                continue
            lines.append(f"{gs:<12s}" + "".join(f"{row.get(h, 0):>{col_w}d}" for h in headers))
    return "\n".join(lines)


def write_results_csv(samples, preds, path):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "gold_label", "pred_intent", "correct",
                    "gold_order_status", "pred_order_status", "order_status_correct",
                    "gold_cancel_scope", "pred_cancel_scope", "cancel_scope_correct",
                    "confidence", "latency_ms", "error", "bond_code", "direction",
                    "price", "amount", "settle_date", "source", "raw_output",
                    "retries", "prompt_tokens", "completion_tokens", "total_tokens"])
        for s, p in zip(samples, preds):
            slots = p.get("slots") or {}
            gold_os, pred_os = s.get("gold_order_status"), p.get("order_status")
            gold_cs, pred_cs = s.get("gold_cancel_scope"), p.get("cancel_scope")
            os_correct = ""
            if s["label"] == "bond_query":
                os_correct = int((pred_os or "all") == (gold_os or "all")
                                 and p["intent"] == "bond_query")
            cs_correct = ""
            if gold_cs is not None:
                cs_correct = int(pred_cs == gold_cs and p["intent"] == "bond_inquiry")
            w.writerow([
                s["text"], s["label"], p["intent"], int(p["intent"] == s["label"]),
                gold_os, pred_os, os_correct,
                gold_cs, pred_cs, cs_correct,
                p["confidence"], p["latency_ms"], p["error"] or "",
                slots.get("bond_code"), slots.get("direction"),
                slots.get("price"), slots.get("amount"), slots.get("settle_date"),
                s["source_sheet"], p["raw_output"],
                p.get("retries", 0), p.get("prompt_tokens", 0),
                p.get("completion_tokens", 0), p.get("total_tokens", 0),
            ])


def write_bad_cases_csv(samples, preds, path):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "gold_label", "pred_intent",
                    "gold_order_status", "pred_order_status",
                    "gold_cancel_scope", "pred_cancel_scope",
                    "err_type", "error", "raw_output", "source"])
        for s, p in zip(samples, preds):
            intent_wrong = p["intent"] != s["label"]
            gold_os, pred_os = s.get("gold_order_status"), p.get("order_status")
            gold_cs, pred_cs = s.get("gold_cancel_scope"), p.get("cancel_scope")
            os_wrong = (s["label"] == "bond_query" and p["intent"] == "bond_query"
                        and (pred_os or "all") != (gold_os or "all"))
            cs_wrong = (gold_cs is not None and p["intent"] == "bond_inquiry"
                        and pred_cs != gold_cs)
            if intent_wrong or os_wrong or cs_wrong or p["error"]:
                err_type = []
                if intent_wrong: err_type.append("intent")
                if os_wrong: err_type.append("order_status")
                if cs_wrong: err_type.append("cancel_scope")
                if p["error"]: err_type.append("parse")
                w.writerow([s["text"], s["label"], p["intent"],
                            gold_os, pred_os, gold_cs, pred_cs,
                            "+".join(err_type),
                            p["error"] or "", p["raw_output"], s["source_sheet"]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--concurrency", type=int, default=config.EVAL_CONCURRENCY)
    parser.add_argument("--output-tag", type=str,
                        default=datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    _print_header("加载测试集")
    samples = load_test_data()
    print(f"  从 {os.path.basename(config.TEST_DATA_PATH)} 读取 {len(samples)} 条")
    retrieval_desc = config.RETRIEVAL_MODE if config.USE_DYNAMIC_FEWSHOT else "static(关闭)"
    voting_desc = f"投票({config.VOTING_TEMPERATURES})" if config.USE_VOTING else "关闭"
    hierarchical_desc = "开启" if config.USE_HIERARCHICAL_ROUTING else "关闭"
    print(f"  召回模式: {retrieval_desc}  投票: {voting_desc}  分层路由: {hierarchical_desc}")
    if args.limit > 0:
        samples = samples[:args.limit]
        print(f"\n[--limit] 仅取前 {len(samples)} 条")

    if args.dry_run:
        _print_header("dry-run 模式: 不调 LLM,只验证数据加载")
        for s in samples[:10]:
            os_str = f" [{s['gold_order_status']}]" if s["label"] == "bond_query" else ""
            print(f"  [{s['label']:14s}]{os_str} {s['text']}")
        return

    _print_header(f"开始评测 (模型: {config.MODEL_ID}, 并发: {args.concurrency})")
    if config.USE_DYNAMIC_FEWSHOT and config.RETRIEVAL_MODE != "static":
        from retriever import get_retriever
        print("  预加载 embedding 模型 ...", flush=True)
        get_retriever().prewarm([s["text"] for s in samples])
        print(f"  ✓ embedding 就绪 ({len(samples)} 条已缓存)", flush=True)
    preds = run_predictions(samples, concurrency=args.concurrency)

    _print_header("计算指标")
    metrics = compute_metrics(samples, preds)
    report = render_report(metrics)
    print(report)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    tag = args.output_tag
    results_path = os.path.join(config.OUTPUT_DIR, f"{tag}_results.csv")
    bad_path = os.path.join(config.OUTPUT_DIR, f"{tag}_bad_cases.csv")
    report_path = os.path.join(config.OUTPUT_DIR, f"{tag}_report.txt")

    write_results_csv(samples, preds, results_path)
    write_bad_cases_csv(samples, preds, bad_path)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n已写入:")
    print(f"  {results_path}")
    print(f"  {bad_path}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()
