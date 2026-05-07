import sys
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict, Any
import itertools  # 用于高效合并词列表

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))
from config.config import parse_args
from src.utils.utils import init_logger

logger = init_logger()



def load_ner_samples(path: Path) -> List[List[Tuple[str, str]]]:
    """加载NER格式数据（每行词\t标签，句子间空行分隔）"""
    sentences = []
    current = []
    
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:  # 空行结束句子
                if current:
                    sentences.append(current)
                    current = []
                continue
                
            parts = line.split("\t")
            if len(parts) < 2 or not parts[0] or not parts[1]:
                logger.warning(f"跳过无效NER行: {line}")
                continue
            current.append((parts[0], parts[1]))
    
    if current:
        sentences.append(current)
    
    logger.info(f"成功加载 {len(sentences)} 句子，共 {sum(len(s) for s in sentences)} 词")
    return sentences

def build_vocab(segments: List[List[str]], vocab_size: int = None, min_freq: int = 1) -> Dict[str, int]:
    """动态构建词汇表（含[PAD]/[UNK]）"""
    base_tokens = ["[PAD]", "[UNK]"]
    all_words = list(itertools.chain.from_iterable(segments))  # 高效合并词列表
    freq = Counter(w for w in all_words if w)  # 过滤空词
    
    # 过滤低频词
    freq = {w: c for w, c in freq.items() if c >= min_freq}
    if not freq:
        raise ValueError("词频过滤后词表为空，请降低 min_freq")

    # 按频率排序
    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    kept_words = []
    cumulative = 0
    for w, c in sorted_freq:
        kept_words.append(w)
        cumulative += c
    
    # 限制词汇表大小
    if vocab_size is not None:
        kept_words = kept_words[:max(0, vocab_size - len(base_tokens))]
    
    return {word: idx for idx, word in enumerate(base_tokens + kept_words)}

def build_ner_label_map(labels: List[str]) -> Dict[str, int]:
    """构建NER标签映射（自动包含O标签）"""
    all_labels = set(labels) | {"O"}
    return {label: idx for idx, label in enumerate(sorted(all_labels))}

def save_mapping(mapping: Dict[Any, Any], path: Path, sort_keys=True):
    """保存字典为文本格式（键值对每行）"""
    path.parent.mkdir(parents=True, exist_ok=True)
    items = sorted(mapping.items()) if sort_keys else mapping.items()
    with open(path, "w", encoding="utf-8") as f:
        for k, v in items:
            f.write(f"{k}\t{v}\n")

def save_ner_dataset(dataset: List[Tuple[List[int], List[int]]], path: Path):
    """保存NER数据集（词ID序列\t标签ID序列）"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for words, labels in dataset:
            f.write(f"{' '.join(map(str, words))}\t{' '.join(map(str, labels))}\n")

def prepare_data(args):
    logger.info("开始NER数据处理流程（使用NER标注格式）")
    
    # 1. 加载数据
    train = load_ner_samples(REPO_ROOT / args["train_path"])
    test = load_ner_samples(REPO_ROOT / args["test_path"])
    
    # 2. 构建词汇表
    all_words = [word for sent in train for word, _ in sent]
    vocab = build_vocab([all_words])  # 传入词列表的列表
    logger.info(f"词汇表大小: {len(vocab)} (含 [PAD]/[UNK])")
    
    # 3. 构建标签映射
    all_labels = [label for sent in train for _, label in sent]
    label_map = build_ner_label_map(all_labels)
    logger.info(f"标签体系: {list(label_map.keys())}")
    
    # 4. 编码参数
    max_len = int(args.get("max_len", 64))
    pad_id = vocab["[PAD]"]
    unk_id = vocab["[UNK]"]
    o_id = label_map["O"]
    
    # 5. 编码句子
    def encode_sentence(sent: List[Tuple[str, str]]) -> Tuple[List[int], List[int]]:
        words, labels = zip(*sent) if sent else ([], [])
        return (
            [vocab.get(w, unk_id) for w in words] + [pad_id] * (max_len - len(words)),
            [label_map.get(l, o_id) for l in labels] + [o_id] * (max_len - len(labels))
        )
    
    # 6. 生成数据集
    train_dataset = [encode_sentence(sent) for sent in train]
    test_dataset = [encode_sentence(sent) for sent in test]
    
    # 7. 保存结果
    save_mapping(vocab, REPO_ROOT / args["lstm_vocab_path"])
    save_mapping(label_map, REPO_ROOT / args["lstm_label_path"])
    save_ner_dataset(train_dataset, REPO_ROOT / args["lstm_train_dataset_path"])
    save_ner_dataset(test_dataset, REPO_ROOT / args["lstm_test_dataset_path"])
    
    logger.info("✅ NER数据准备完成！")
    logger.info(f"训练样本数: {len(train_dataset)}, 测试样本数: {len(test_dataset)}")
    return vocab, label_map, train_dataset, test_dataset

def main():
    args = parse_args()
    logger.info("解析NER参数: %s", args)
    prepare_data(args)
    logger.info("✅ NER数据处理完成")

if __name__ == "__main__":
    main()