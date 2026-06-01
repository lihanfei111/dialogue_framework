# -*- coding: utf-8 -*-
"""
一次性脚本:把 fewshot_pool.json 里所有 text 用 bge-small-zh-v1.5 编码成向量,
保存为 fewshot_pool_vectors.npy。后续 retriever 启动只读这个 .npy,无重算开销。

加载优先级(从高到低):
  1. --model-path 显式指定本地路径
  2. config.EMBEDDING_LOCAL_PATH(ModelScope 本地缓存,目录存在时自动使用)
  3. --source modelscope(在线从魔搭下载)
  4. HuggingFace(默认兜底)

使用:
    python _build_vectors.py                      # 自动用本地 ModelScope 缓存(推荐)
    python _build_vectors.py --source modelscope  # 强制从魔搭下载
    python _build_vectors.py --model-path /path/to/local/model  # 指定任意本地路径
"""
import argparse
import json
import os
import sys

import numpy as np

import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None,
                        help="显式指定本地模型路径,跳过自动检测")
    parser.add_argument("--source", choices=["hf", "modelscope"], default="hf",
                        help="无本地缓存时的下载来源。hf=HuggingFace;modelscope=阿里魔搭(国内推荐)")
    args = parser.parse_args()

    # 优先级: 显式 --model-path > EMBEDDING_LOCAL_PATH(本地缓存) > 在线下载
    if args.model_path:
        model_id = args.model_path
        print(f"从指定路径加载模型: {model_id}")
    elif config.EMBEDDING_LOCAL_PATH and os.path.isdir(config.EMBEDDING_LOCAL_PATH):
        model_id = config.EMBEDDING_LOCAL_PATH
        print(f"从本地 ModelScope 缓存加载: {model_id}")
    elif args.source == "modelscope":
        try:
            from modelscope import snapshot_download
        except ImportError:
            print("缺 modelscope: pip install modelscope"); sys.exit(1)
        print("从 modelscope 下载 bge-small-zh-v1.5(首次较慢)...")
        model_id = snapshot_download(config.EMBEDDING_MODEL_NAME)
    else:
        model_id = config.EMBEDDING_MODEL_NAME
        print(f"从 HuggingFace 加载: {model_id}")
        print("(国内若 403,设环境变量: set HF_ENDPOINT=https://hf-mirror.com)")

    from sentence_transformers import SentenceTransformer
    print("加载模型 ...")
    model = SentenceTransformer(model_id)
    print(f"  ✓ 嵌入维度 {model.get_sentence_embedding_dimension()}")

    # 加载召回库
    pool = json.load(open(config.FEWSHOT_POOL_PATH, encoding="utf-8"))
    texts = [e["text"] for e in pool]
    print(f"编码 {len(texts)} 条召回库 ...")
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    print(f"  ✓ 输出 shape={vectors.shape}, dtype={vectors.dtype}")

    np.save(config.FEWSHOT_VECTORS_PATH, vectors.astype(np.float32))
    print(f"\n已保存 {config.FEWSHOT_VECTORS_PATH}")
    print(f"  文件大小 ≈ {os.path.getsize(config.FEWSHOT_VECTORS_PATH) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
