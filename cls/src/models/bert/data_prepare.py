from pathlib import Path
import sys
from collections import Counter

# 添加仓库根目录到 sys.path（当前文件上三级）
sys.path.append(str(Path(__file__).resolve().parents[3]))
from config.config import parse_args


def prepare_data(args):
    """
    读取训练集与测试集，构建词表与数据集。
    返回: word2idx, idx2word, label2idx, idx2label, train_dataset, test_dataset
    其中 train_dataset/test_dataset = [(input_ids, label_id), ...]
    """
    repo_root = Path(__file__).resolve().parents[3]

    def to_abs(path):  # 相对路径转绝对路径
        path = Path(path)
        return path if path.is_absolute() else (repo_root / path).resolve()

    def read_samples(path):
        """读取样本 (text, label)"""
        samples = []
        with path.open("r", encoding="utf-8") as f:
            for line in map(str.strip, f):
                if not line:
                    continue
                # 优先按 tab 分割，否则按最后一个空格分割
                if "\t" in line:
                    text, label = line.split("\t", 1)
                else:
                    parts = line.rsplit(maxsplit=1)
                    if len(parts) != 2:
                        continue
                    text, label = parts
                samples.append((text.strip(), label.strip()))
        return samples

    # 获取路径并验证
    train_path, test_path = map(to_abs, (args["train_file"], args["test_file"]))
    for p in [train_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"文件不存在: {p}")

    max_len = int(args.get("max_len", 128))
    vocab_size = int(args.get("vocab_size", 0))

    # 读取样本
    train_samples = read_samples(train_path)
    test_samples = read_samples(test_path)

    # 标签映射
    labels = sorted({lab for _, lab in train_samples})
    label2idx = {lab: i for i, lab in enumerate(labels)}
    idx2label = {i: lab for lab, i in label2idx.items()}

    # 构建词表（按训练语料频次）
    counter = Counter(tok for text, _ in train_samples for tok in text.split())
    vocab_tokens = ["[PAD]", "[UNK]"]

    # 如果 vocab_size == 0 则使用全部词汇；否则取前 vocab_size 个（包含特殊符号），但不超过语料实际词汇量
    most_common_tokens = [tok for tok, _ in counter.most_common()]
    if vocab_size <= 0:
        top_tokens = most_common_tokens
    else:
        # 计算除特殊符号外还需要的词数，确保不超过实际词汇数量
        need = max(0, vocab_size - len(vocab_tokens))
        top_tokens = most_common_tokens[:need]

    vocab_tokens += top_tokens
    word2idx = {tok: i for i, tok in enumerate(vocab_tokens)}
    idx2word = {i: tok for tok, i in word2idx.items()}

    def encode(text):
        """文本转 id 序列（截断/填充）"""
        ids = [word2idx.get(t, word2idx["[UNK]"]) for t in text.split()]
        return ids[:max_len] + [word2idx["[PAD]"]] * max(0, max_len - len(ids))

    def build_dataset(samples):
        """构建数据集"""
        return [
            (encode(text), label2idx[lab])
            for text, lab in samples if lab in label2idx
        ]

    return (
        word2idx, idx2word,
        label2idx, idx2label,
        build_dataset(train_samples),
        build_dataset(test_samples),
    )


def main():
    args = parse_args()
    print(args)
    print(prepare_data(args))


if __name__ == "__main__":
    main()