import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
from transformers import BertTokenizer  # 新增BERT tokenizer

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

    logger.info(
        f"成功加载 {len(sentences)} 句子，共 {sum(len(s) for s in sentences)} 词"
    )
    return sentences


def build_ner_label_map(labels: List[str]) -> Dict[str, int]:
    """构建NER标签映射（自动包含B/I标签）"""
    all_labels = set()
    all_labels.add("O")  # 添加O标签
    for label in labels:
        if label != "O":
            all_labels.add(label)  # 添加B-标签
            # 为B-标签生成对应的I-标签
            if label.startswith("B-"):
                all_labels.add("I-" + label[2:])
    return {label: idx for idx, label in enumerate(sorted(all_labels))}


def save_mapping(mapping: Dict[Any, Any], path: Path, sort_keys=True):
    """保存字典为文本格式（键值对每行）"""
    path.parent.mkdir(parents=True, exist_ok=True)
    items = sorted(mapping.items()) if sort_keys else mapping.items()
    with open(path, "w", encoding="utf-8") as f:
        for k, v in items:
            f.write(f"{k}\t{v}\n")


def save_ner_dataset(dataset: List[Tuple[List[int], List[int]]], path: Path):
    """保存NER数据集（token ID序列\t标签ID序列）"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for tokens, labels in dataset:
            f.write(f"{' '.join(map(str, tokens))}\t{' '.join(map(str, labels))}\n")


def prepare_data(args):
    logger.info("开始BERT NER数据处理流程（使用NER标注格式）")

    # 1. 加载数据
    train = load_ner_samples(REPO_ROOT / args["train_path"])
    test = load_ner_samples(REPO_ROOT / args["test_path"])

    # 2. 构建标签映射（自动包含B/I标签）
    all_labels = [label for sent in train for _, label in sent]
    label_map = build_ner_label_map(all_labels)
    logger.info(f"标签体系: {list(label_map.keys())}")

    # 3. 加载BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(REPO_ROOT/args["bert_model_name_or_path"])
    logger.info(f"加载BERT tokenizer: {REPO_ROOT/args['bert_model_name_or_path']}")

    # 4. 设置BERT输入长度（包含[CLS]/[SEP]）
    max_len = int(args.get("max_len", 64))
    logger.info(f"BERT输入最大长度: {max_len} (含[CLS]/[SEP])")

    # 5. 编码句子（BERT专用）
    def encode_sentence(sent: List[Tuple[str, str]]) -> Tuple[List[int], List[int]]:
        tokens = []
        labels = []

        # 处理每个词的分词和标签对齐
        for word, label in sent:
            # 分词（不包含特殊token）
            word_tokens = tokenizer.tokenize(word)

            if label != "O":
                # 添加B-标签（第一个子词）
                tokens.extend(word_tokens)
                labels.append(label)
                # 添加I-标签（后续子词）
                if len(word_tokens) > 1:
                    labels.extend(["I-" + label[2:]] * (len(word_tokens) - 1))
            else:
                # O标签直接扩展
                tokens.extend(word_tokens)
                labels.extend(["O"] * len(word_tokens))

        # 添加[CLS]和[SEP]特殊token
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        labels = ["O"] + labels + ["O"]

        # 截断或填充到max_len
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
            labels = labels[:max_len]
        else:
            padding = max_len - len(tokens)
            tokens.extend(["[PAD]"] * padding)
            labels.extend(["O"] * padding)

        # 转换为ID
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        label_ids = [label_map[l] for l in labels]

        return token_ids, label_ids

    # 6. 生成数据集
    train_dataset = [encode_sentence(sent) for sent in train]
    test_dataset = [encode_sentence(sent) for sent in test]

    # 7. 保存结果
    save_mapping(label_map, REPO_ROOT / args["bert_label_path"])
    save_ner_dataset(train_dataset, REPO_ROOT / args["bert_train_dataset_path"])
    save_ner_dataset(test_dataset, REPO_ROOT / args["bert_test_dataset_path"])

    logger.info("✅ BERT NER数据准备完成！")
    logger.info(f"训练样本数: {len(train_dataset)}, 测试样本数: {len(test_dataset)}")
    logger.info(
        f"保存路径: {args['bert_label_path']}, {args['bert_train_dataset_path']}"
    )
    return label_map, train_dataset, test_dataset


def main():
    args = parse_args()
    logger.info("解析BERT NER参数: %s", args)
    prepare_data(args)
    logger.info("✅ BERT NER数据处理完成")


if __name__ == "__main__":
    main()
