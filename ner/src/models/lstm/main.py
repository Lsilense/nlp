import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Dict, Any
from pathlib import Path
from tqdm import tqdm
import sys
import os

# 添加项目根目录
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

from src.models.lstm.model_lstm import LSTM_CRF
from config.config import parse_args
from src.utils.utils import init_logger

logger = init_logger()


class NERDataset(Dataset):
    """NER数据集，处理序列标注任务"""

    def __init__(self, data: List[Tuple[List[int], List[int]]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(
    batch: List[Tuple[List[int], List[int]]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """动态填充批次数据并生成mask"""
    words, tags = zip(*batch)

    # 填充序列
    padded_words = pad_sequence(
        [torch.tensor(w, dtype=torch.long) for w in words],
        batch_first=True,
        padding_value=0,
    )

    padded_tags = pad_sequence(
        [torch.tensor(t, dtype=torch.long) for t in tags],
        batch_first=True,
        padding_value=0,
    )

    # 创建mask (1=有效, 0=填充)
    mask = padded_words != 0

    return padded_words, padded_tags, mask


def load_vocab(vocab_path: str) -> Dict[str, int]:
    """加载词汇表"""
    vocab = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            word, idx = line.strip().split("\t")
            vocab[word] = int(idx)
    logger.info(f"加载词汇表: {len(vocab)} 个词")
    return vocab


def load_label_map(label_map_path: str) -> Dict[str, int]:
    """加载标签映射"""
    label_map = {}
    with open(label_map_path, "r", encoding="utf-8") as f:
        for line in f:
            label, idx = line.strip().split("\t")
            label_map[label] = int(idx)
    logger.info(f"加载标签映射: {len(label_map)} 个标签")
    return label_map


def load_dataset(dataset_path: str) -> List[Tuple[List[int], List[int]]]:
    """加载NER数据集"""
    dataset = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                words_str, labels_str = line.split("\t")
                dataset.append(
                    (
                        list(map(int, words_str.split())),
                        list(map(int, labels_str.split())),
                    )
                )
            except Exception as e:
                logger.error(f"解析第 {line_num} 行出错: {line} -> {e}")
                raise ValueError(f"数据格式错误，行 {line_num}: {line}")

    logger.info(f"加载数据集: {len(dataset)} 条样本")
    return dataset


def evaluate(
    model: nn.Module, data_loader: DataLoader, device: str
) -> Tuple[float, float]:
    """评估模型性能 (损失和token准确率)"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for words, tags, mask in data_loader:
            words, tags, mask = words.to(device), tags.to(device), mask.to(device)
            loss = model(words, tags, mask)
            total_loss += loss.item()

            # 计算token准确率
            emissions = model(words, mask=mask)
            predictions = model.crf.decode(emissions, mask=mask)  # 返回列表

            # 修复：正确处理不同长度的预测结果
            for i, (pred, true_tag, m) in enumerate(zip(predictions, tags, mask)):
                # 获取实际长度（非填充部分）
                actual_len = m.sum().item()
                # 只比较非填充部分
                pred_tensor = torch.tensor(pred, dtype=torch.long, device=device)
                correct += (pred_tensor == true_tag[:actual_len]).sum().item()
                total += actual_len

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_dataset: NERDataset,
    test_dataset: NERDataset,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    model_save_path: str,  # 新增：模型保存路径
) -> None:
    """训练LSTM-CRF模型，使用tqdm显示训练进度，并保存最佳模型"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)

    logger.info(f"开始训练模型 (设备: {device})")

    # 初始化最佳准确率
    best_acc = 0.0

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0.0
        model_save_path = Path(REPO_ROOT / model_save_path)
        os.makedirs(model_save_path, exist_ok=True)
        final_model_path = model_save_path / "ner_lstm_final.pth"
        best_model_path = model_save_path / "ner_lstm_best.pth"

        # 使用tqdm包装train_loader显示进度条
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False
        )

        for words, tags, mask in progress_bar:
            words, tags, mask = words.to(device), tags.to(device), mask.to(device)

            optimizer.zero_grad()
            loss = model(words, tags, mask)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 实时更新进度条上的loss信息
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # 评估阶段
        train_loss = total_loss / len(train_loader)
        test_loss, test_acc = evaluate(model, test_loader, device)

        # 在日志中记录完整epoch结果
        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.4f}"
        )

        # 保存最佳模型（测试准确率最高时）
        if test_acc > best_acc:
            best_acc = test_acc

            torch.save(model.state_dict(), best_model_path)
            logger.info(f"最佳模型已保存到: {best_model_path} (准确率: {test_acc:.4f})")

    # 保存最终模型
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"最后一轮模型已保存到: {final_model_path} (准确率: {test_acc:.4f})")

    logger.info(f"训练完成，最佳准确率: {best_acc:.4f}")


def main(args: Dict[str, Any]):
    """主函数，执行训练和预测流程"""
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 分离加载词汇表、标签映射和数据集
    vocab = load_vocab(REPO_ROOT / args["lstm_vocab_path"])
    label_map = load_label_map(REPO_ROOT / args["lstm_label_path"])

    train_data = load_dataset(REPO_ROOT / args["lstm_train_dataset_path"])
    test_data = load_dataset(REPO_ROOT / args["lstm_test_dataset_path"])

    # 创建数据集
    train_dataset = NERDataset(train_data)
    test_dataset = NERDataset(test_data)

    # 创建模型
    model = LSTM_CRF(
        vocab_size=len(vocab),
        tag_size=len(label_map),
        embed_dim=args.get("embed_dim", 100),
        hidden_dim=args.get("hidden_dim", 200),
    )

    # 训练模型
    train_model(
        model,
        train_dataset,
        test_dataset,
        epochs=args["epochs"],
        batch_size=args["batch_size"],
        lr=args["lr_lstm"],
        device=device,
        model_save_path=args["lstm_model_path"],  # 传入保存路径
    )

    # ====== 新增：推理示例 ======
    # 加载最佳模型进行推理
    model = LSTM_CRF(
        vocab_size=len(vocab),
        tag_size=len(label_map),
        embed_dim=args.get("embed_dim", 100),
        hidden_dim=args.get("hidden_dim", 200),
    )
    model.load_state_dict(
        torch.load(
            REPO_ROOT / args["lstm_model_path"] / "ner_lstm_best.pth",
            map_location=device,
            weights_only=True,
        )
    )
    model = model.to(device)
    model.eval()

    # 测试句子：北京是中国的首都。
    test_sentence = "北京是中国的首都。".strip()  # 需要分词（实际应用建议用分词工具）
    test_sentence = list(test_sentence)  # 简单按字符切分
    print("\n=== 推理结果 ===")
    print(f"输入句子: {' '.join(test_sentence)}")
    result = model.predict(test_sentence, vocab, label_map, device=device)
    print("预测结果:", result)
    print("================")

    # 可视化结果
    for word, tag in result:
        print(f"{word} -> {tag}")


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    # 执行主流程
    main(args)
