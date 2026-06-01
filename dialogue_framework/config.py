"""
集中配置。要换 API、Redis、模型、文件路径,只改这里。
"""
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# ============ LLM API ============
API_KEY = "sk-t9yrWr504E0euAfWuMm7cA"
BASE_URL = "http://192.168.17.9:1234/v1"
MODEL_ID = "qwen3.6-35b-a3b"
TEMPERATURE = 0.0
MAX_TOKENS = 2048
TIMEOUT = 120
MAX_RETRIES = 2
DEBUG_LLM = True   # True 时每次 LLM 调用后打印原始响应摘要(排查思考/解析问题)

# ============ Session 状态存储(Redis) ============
# Redis 连接 URL。格式:redis://[:password@]host:port/db
# 生产环境从环境变量读更安全:os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_URL = "redis://localhost:6379/0"

# session 状态在 Redis 里的过期时间(秒)。
# 设为 None 表示永不过期;默认 24 小时,跨日新会话天然隔离。
SESSION_TTL_SECONDS = 24 * 3600

# Redis 中 session key 的前缀,方便和别的业务共用一个 Redis 时区分
SESSION_KEY_PREFIX = "ficc_dialogue:session:"

# ============ 上下文窗口 ============
# recent_turns 最多保留多少轮(防止 state 无限膨胀)
MAX_RECENT_TURNS = 6

# ============ 意图识别评测配置(给 eval_intent.py 用) ============
# Excel 路径(相对于 config.py 所在目录)
EXCEL_PATH = os.path.join(_THIS_DIR, "FICC机器人优化迭代清单.xlsx")
OUTPUT_DIR = os.path.join(_THIS_DIR, "outputs")

# 5 个意图标签
INTENT_LABELS = ["bond_inquiry", "bond_query", "bond_selection", "chat", "transfer_human"]

# bond_query 的细分订单状态(3 种,对应 excel "匹配订单状态")
# all=所有状态;pending=进行中/在挂;completed=已成交/已完成(交易语境同义,合并)
ORDER_STATUS_LABELS = ["all", "pending", "completed"]

# bond_inquiry 撤单细分(对应 excel 撤单文案的 全撤/指定撤)
CANCEL_SCOPE_LABELS = ["all", "specified"]

# 评测并发数
EVAL_CONCURRENCY = 5

# ============ 动态 few-shot 召回配置(embedding 版) ============
FEWSHOT_POOL_PATH = os.path.join(_THIS_DIR, "fewshot_pool.json")
FEWSHOT_VECTORS_PATH = os.path.join(_THIS_DIR, "fewshot_pool_vectors.npy")
# Embedding 模型(开源、中文优化、轻量;Apache 2.0)
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
# ModelScope 本地缓存路径;不为空且目录存在时优先使用,跳过网络下载
EMBEDDING_LOCAL_PATH = r"C:\Users\HUAWEI\.cache\modelscope\hub\models\BAAI\bge-small-zh-v1___5"
# CachedEmbedder 磁盘缓存路径（JSONL，append-only）
FEWSHOT_EMBED_CACHE_PATH = os.path.join(_THIS_DIR, "fewshot_embed_cache.jsonl")
# 最终注入 prompt 的示例条数(召回后按意图均衡交织取这么多条)
FEWSHOT_K = 6
USE_DYNAMIC_FEWSHOT = True

# ============ 消融实验开关 ============
# 3 个开关完全正交,可任意组合(2×3×2=12 种)。全默认 = v6 baseline。
USE_HIERARCHICAL_ROUTING = False  # True:主意图+子分类拆成 2 次 LLM 调用
RETRIEVAL_MODE = "bge"            # "bge" 单路 / "multi_route" 多路 / "static" 静态兜底
USE_VOTING = False                # True:同模型不同 temperature 3 次调用 majority vote
VOTING_TEMPERATURES = [0.0, 0.3, 0.6]  # 投票调用的 3 个温度

# ============ 测试集(给 eval_intent.py 用) ============
TEST_DATA_PATH = os.path.join(_THIS_DIR, "test_data.jsonl")

# 每个 sheet 怎么读、对应哪个意图标签
SHEET_CONFIGS = [
    {"sheet": "闲聊收集文案", "intent": "chat",         "columns": [0, 2], "skip_first_row": False},
    {"sheet": "报价文案",     "intent": "bond_inquiry", "columns": [0],    "skip_first_row": False},
    {"sheet": "撤单语料",     "intent": "bond_inquiry", "columns": [0],    "skip_first_row": True},
    {"sheet": "撤单文案",     "intent": "bond_inquiry", "columns": [0],    "skip_first_row": True},
    {"sheet": "查询订单语料", "intent": "bond_query",   "columns": [0],    "skip_first_row": True},
    # 后续补 bond_selection 语料时在这里加
]
