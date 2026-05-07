import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json

# 🔥 使用 transformers 提供的 BertTokenizer
from transformers import BertTokenizer

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


def save_mapping(mapping: Dict[Any, Any], path: Path):
    """保存字典为文本格式：标签\t索引"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        # 按照索引值排序后保存
        sorted_items = sorted(mapping.items(), key=lambda x: x[1])
        for label, idx in sorted_items:
            f.write(f"{label}\t{idx}\n")


def save_dataset(dataset: List[Dict[str, Any]], path: Path):
    """保存为 JSONL 格式（每行一个 dict），便于后续读取"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def prepare_data(args):
    logger.info("正在准备 BERT 所需的数据...")

    # 路径配置
    train_file = REPO_ROOT / args["train_path"]
    test_file = REPO_ROOT / args["test_path"]
    model_name_or_path = REPO_ROOT / args["bert_model_name_or_path"]  # 可配置
    max_length = int(args.get("max_len", 64))

    # 加载 tokenizer
    logger.info(f"加载 BERT Tokenizer: {model_name_or_path}")
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)

    # 加载训练集和测试集
    train_samples = load_samples(train_file)
    test_samples = load_samples(test_file)

    # 构建标签映射（仅限训练集中出现的标签）
    unique_labels = sorted({label for _, label in train_samples})
    labels = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    logger.info(f"发现 {len(labels)} 个唯一标签: {list(labels.keys())}")

    def encode_sample(text: str, label: str, max_len: int):
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # 添加 [CLS] 和 [SEP]
            max_length=max_len,
            padding="max_length",  # 补全到 max_length
            truncation=True,  # 截断超过长度的部分
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors=None,  # 返回 Python list
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "token_type_ids": encoded["token_type_ids"],
            "label_id": labels[label],
        }

    # 编码训练集
    logger.info("正在对训练集进行编码...")
    train_dataset = []
    skipped_train = 0
    for text, label in train_samples:
        if label not in labels:
            skipped_train += 1
            continue
        encoded = encode_sample(text, label, max_length)
        train_dataset.append(encoded)
    if skipped_train:
        logger.warning(f"训练集中跳过 {skipped_train} 条未知标签样本")

    # 编码测试集
    logger.info("正在对测试集进行编码...")
    test_dataset = []
    skipped_test = 0
    for text, label in test_samples:
        if label not in labels:
            skipped_test += 1
            continue
        encoded = encode_sample(text, label, max_length)
        test_dataset.append(encoded)
    if skipped_test:
        logger.warning(f"测试集中跳过 {skipped_test} 条未知标签样本")

    # 保存结果
    save_mapping(labels, REPO_ROOT / args["bert_label_path"])  # 保存 label_to_id
    save_dataset(train_dataset, REPO_ROOT / args["bert_train_dataset_path"])
    save_dataset(test_dataset, REPO_ROOT / args["bert_test_dataset_path"])

    logger.info(f"✅ 数据准备完成！")
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"测试集大小: {len(test_dataset)}")
    logger.info(f"序列长度: {max_length}")
    logger.info(f"使用的 BERT 模型: {model_name_or_path}")

    return labels, train_dataset, test_dataset


def main():
    args = parse_args()
    logger.info("解析参数: %s", args)

    result = prepare_data(args)

    logger.info("✅ BERT 数据预处理已完成！")
    logger.info(f"标签集合: {list(result[0].keys())}")
    if len(result[1]) > 0:
        logger.info(f"前两个训练样本 input_ids 示例: {result[1][:2]}")


if __name__ == "__main__":
    main()