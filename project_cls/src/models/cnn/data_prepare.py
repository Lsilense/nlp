import sys
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict, Any
import jieba  # 🔥 使用 jieba 替代 LTP

# 添加项目根目录
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))
from config.config import parse_args
from src.utils.utils import init_logger

logger = init_logger()


def load_samples(path: Path) -> List[Tuple[str, str]]:
    """加载文本-标签样本，支持 tab 或 空格分割"""
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                text, label = line.split("\t", 1)
            else:
                parts = line.rsplit(maxsplit=1)
                if len(parts) != 2:
                    logger.warning(f"跳过无效行: {line}")
                    continue
                text, label = parts
            samples.append((text.strip(), label.strip()))
    return samples


def build_vocab(segments: List[List[str]], vocab_size: int = None, min_freq: int = 2, coverage_ratio: float = 0.98) -> Dict[str, int]:
    """
    动态构建词汇表：
      1) 去除低频词（频率 < min_freq）
      2) 根据 coverage_ratio 动态截断，以覆盖绝大部分语料的词（比如 98%）
      3) 如果用户指定 vocab_size，则作为最大上限
    """
    base_tokens = ["[PAD]", "[UNK]"]

    # 统计词频
    freq = Counter(word for seg_list in segments for word in seg_list)

    # 步骤 1：去除低频词
    freq = Counter({w: c for w, c in freq.items() if c >= min_freq})
    if not freq:
        raise ValueError("词频过滤后词表为空，请降低 min_freq。")

    # 步骤 2：根据 coverage_ratio 动态截断
    total = sum(freq.values())
    sorted_items = freq.most_common()

    kept_words = []
    cumulative = 0
    for w, c in sorted_items:
        kept_words.append(w)
        cumulative += c
        if cumulative / total >= coverage_ratio:
            break

    # 若 vocab_size 设置，则再限制最大长度
    if vocab_size:
        kept_words = kept_words[:max(0, vocab_size - len(base_tokens))]

    final_vocab = base_tokens + kept_words
    return {word: idx for idx, word in enumerate(final_vocab)}



def save_mapping(mapping: Dict[Any, Any], path: Path, sort_keys=True):
    """保存字典为文本格式：key\tvalue"""
    path.parent.mkdir(parents=True, exist_ok=True)
    items = sorted(mapping.items()) if sort_keys else mapping.items()
    with open(path, "w", encoding="utf-8") as f:
        for k, v in items:
            f.write(f"{k}\t{v}\n")


def save_dataset(dataset: List[Tuple[List[int], int]], path: Path):
    """保存数据集为文本格式：id序列\tlabel_id"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for input_ids, label_id in dataset:
            ids_str = " ".join(map(str, input_ids))
            f.write(f"{ids_str}\t{label_id}\n")


def segment_all_texts_jieba(texts: List[str], use_fast=True, batch_log=1000) -> List[List[str]]:
    if use_fast:
        try:
            jieba.enable_parallel()
            logger.info("✅ 已启用 jieba 并行分词模式")
        except:
            logger.info("⚠️ 并行分词不可用（Windows 不支持），使用单线程")

    total = len(texts)
    logger.info(f"开始使用 jieba 分词，共 {total} 条文本...")

    all_segments = []
    invalid_count = 0
    for i, text in enumerate(texts):
        words = list(jieba.cut(text))
        
        # 过滤：只保留去除空白后非空的词
        valid_words = [w for w in words if w.strip()]
        invalid_count += len(words) - len(valid_words)
        
        all_segments.append(valid_words)

        if (i + 1) % batch_log == 0 or i == total - 1:
            logger.info(f"已完成 {i + 1}/{total} 条分词")

    if invalid_count > 0:
        logger.warning(f"🗑️ 分词过程中共过滤 {invalid_count} 个空/无效 token")

    return all_segments


def prepare_data(args):
    logger.info("正在使用 jieba 进行中文分词...")

    train_file = REPO_ROOT / args["train_path"]
    test_file = REPO_ROOT / args["test_path"]

    train_samples = load_samples(train_file)
    test_samples = load_samples(test_file)

    # 提取标签映射
    unique_labels = sorted({label for _, label in train_samples})
    labels = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    logger.info(f"发现 {len(labels)} 个唯一标签: {list(labels.keys())}")

    # 提取所有文本进行分词
    train_texts = [text for text, _ in train_samples]
    test_texts = [text for text, _ in test_samples]

    # 🔥 使用 jieba 批量分词
    logger.info("正在进行 jieba 分词...")
    train_segments = segment_all_texts_jieba(train_texts)
    test_segments = segment_all_texts_jieba(test_texts)

    # 构建词汇表（仅训练集）
    vocabs = build_vocab(train_segments)
    logger.info(f"词汇表大小: {len(vocabs)}")

    # 编码参数
    max_len = int(args.get("max_len", 64))
    unk_id = vocabs["[UNK]"]
    pad_id = vocabs["[PAD]"]

    def encode(words: List[str]) -> List[int]:
        ids = [vocabs.get(w, unk_id) for w in words]
        return ids[:max_len] + [pad_id] * (max_len - len(ids))

    # 编码数据集
    train_dataset = [(encode(seg), labels[lbl]) for seg, (_, lbl) in zip(train_segments, train_samples)]
    test_dataset = [
        (encode(seg), labels[lbl])
        for seg, (_, lbl) in zip(test_segments, test_samples)
        if lbl in labels
    ]

    # 保存结果
    save_mapping(vocabs, REPO_ROOT / args["cnn_vocab_path"])
    save_mapping(labels, REPO_ROOT / args["cnn_label_path"])
    save_dataset(train_dataset, REPO_ROOT / args["cnn_train_dataset_path"])
    save_dataset(test_dataset, REPO_ROOT / args["cnn_test_dataset_path"])

    logger.info("✅ 数据准备完成（使用 jieba）")
    return vocabs, labels, train_dataset, test_dataset


def main():
    args = parse_args()
    logger.info("解析参数: %s", args)

    result = prepare_data(args)

    logger.info("✅ 数据准备完成！")
    logger.info(f"标签集合: {list(result[1].keys())}")
    logger.info(f"前两个训练样本: {result[2][:2]}")


if __name__ == "__main__":
    main()