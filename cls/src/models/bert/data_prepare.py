from pathlib import Path
import sys
from collections import Counter
# 确保仓库根目录可访问
sys.path.append(str(Path(__file__).resolve().parents[3]))
from config.config import parse_args


def prepare_data(args):
    """
    读取训练集与测试集，构建词表与数据集。
    返回:
        word2idx, idx2word, label2idx, idx2label, train_dataset, test_dataset
    """
    repo_root = Path(__file__).resolve().parents[3]
    to_abs = lambda p: Path(p).resolve() if Path(p).is_absolute() else (repo_root / p).resolve()

    # ---------- 工具函数 ----------
    def read_samples(path: Path):
        """读取样本: 返回 [(text, label), ...]"""
        samples = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 按 tab 或最后一个空格分割
                if "\t" in line:
                    text, label = line.split("\t", 1)
                else:
                    parts = line.rsplit(maxsplit=1)
                    if len(parts) != 2:
                        continue
                    text, label = parts
                samples.append((text.strip(), label.strip()))
        return samples

    def encode(text: str):
        """文本转 id 序列（截断或填充）"""
        ids = [word2idx.get(ch, word2idx["[UNK]"]) for ch in text]
        return ids[:max_len] + [word2idx["[PAD]"]] * max(0, max_len - len(ids))

    def build_dataset(samples):
        """构建数据集 (input_ids, label_id)"""
        return [(encode(text), label2idx[label]) for text, label in samples if label in label2idx]

    # ---------- 读取数据 ----------
    train_path, test_path = map(to_abs, (args["train_file"], args["test_file"]))
    for p in (train_path, test_path):
        if not p.exists():
            raise FileNotFoundError(f"文件不存在: {p}")

    train_samples, test_samples = map(read_samples, (train_path, test_path))

    # ---------- 构建标签映射 ----------
    labels = sorted({label for _, label in train_samples})
    label2idx = {lab: i for i, lab in enumerate(labels)}
    idx2label = {i: lab for lab, i in label2idx.items()}

    # ---------- 构建词表 ----------
    max_len = int(args.get("max_len", 64))
    vocab_size = int(args.get("vocab_size", 0))

    token_freq = Counter(ch for text, _ in train_samples for ch in text)
    base_tokens = ["[PAD]", "[UNK]"]

    most_common = [tok for tok, _ in token_freq.most_common()]
    need = max(0, vocab_size - len(base_tokens)) if vocab_size > 0 else len(most_common)
    vocab_tokens = base_tokens + most_common[:need]

    word2idx = {tok: i for i, tok in enumerate(vocab_tokens)}
    idx2word = {i: tok for tok, i in word2idx.items()}

    # ---------- 返回结果 ----------
    return (
        word2idx, idx2word,
        label2idx, idx2label,
        build_dataset(train_samples),
        build_dataset(test_samples),
    )


def main():
    args = parse_args()
    print(args)
    results = prepare_data(args)
    print("数据准备完成，示例：", results[4][:2])  # 打印前两个训练样本示例


if __name__ == "__main__":
    
    main()
