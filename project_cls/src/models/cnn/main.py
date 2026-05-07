from pathlib import Path
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# 添加项目根目录
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

from config.config import parse_args
from src.utils.utils import init_logger
from src.models.cnn.model_cnn import TextCNN, TextClassifier

logger = init_logger()


def collate_fn(batch):
    """动态填充批次数据"""
    input_ids_list, labels_list = zip(*batch)
    padded_input_ids = pad_sequence(
        [torch.tensor(ids, dtype=torch.long) for ids in input_ids_list],
        batch_first=True,
        padding_value=0,
    )
    labels = torch.tensor(labels_list, dtype=torch.long)
    return padded_input_ids, labels


def load_dataset(path):
    """加载预处理的 dataset 文件"""
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ids_str, label_str = line.split("\t")
                input_ids = [int(x) for x in ids_str.split()]
                label_id = int(label_str)
                dataset.append((input_ids, label_id))
            except Exception as e:
                logger.error(f"解析第 {line_num} 行出错: '{line}' -> {e}")
                raise ValueError(f"数据格式错误，行 {line_num}: {line}")
    logger.info(f"从 {path} 加载了 {len(dataset)} 条样本")
    return dataset


def load_word2idx(path):
    """加载 word2idx 映射表，并生成对应的 idx2word"""
    word2idx = {}
    idx2word = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid line in word2idx: {line}")
            word, idx_str = parts
            idx = int(idx_str)
            word2idx[word] = idx
            idx2word[idx] = word

    logger.info(f"词汇表大小: {len(word2idx)} (来自 {path})")
    return word2idx, idx2word


def load_label2idx(path):
    """加载 label2idx 映射表，并生成对应的 idx2label"""
    label2idx = {}
    idx2label = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid line in label2idx: {line}")
            label, idx_str = parts
            idx = int(idx_str)
            label2idx[label] = idx
            idx2label[idx] = label

    logger.info(f"标签数量: {len(label2idx)} (来自 {path})")
    return label2idx, idx2label


def evaluate(model, data_loader, device, criterion):
    """评估模型准确率和平均损失"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / len(data_loader)
    return avg_loss, accuracy


def save_model(model, path):
    """保存模型状态字典"""
    torch.save(model.state_dict(), path)
    logger.info(f"模型已保存至: {path}")


def ensure_min_length(dataset, min_length=5):
    """确保数据集中所有序列至少达到最小长度"""
    processed_dataset = []
    for input_ids, label in dataset:
        if len(input_ids) < min_length:
            # 填充到最小长度
            padded_ids = input_ids + [0] * (min_length - len(input_ids))
            processed_dataset.append((padded_ids, label))
        else:
            processed_dataset.append((input_ids, label))
    return processed_dataset


def main(args):
    # ----------------------------
    # 设备设置
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # ----------------------------
    # 加载词汇表
    # ----------------------------
    vocab_path = REPO_ROOT / args["cnn_vocab_path"]
    label_path = REPO_ROOT / args["cnn_label_path"]
    word2idx, idx2word = load_word2idx(vocab_path)
    label2idx, idx2label = load_label2idx(label_path)
    vocab_size = len(word2idx)
    logger.info(f"词汇表大小: {vocab_size}, 标签数量: {len(label2idx)}")

    # ----------------------------
    # 加载数据集
    # ----------------------------
    train_data_path = REPO_ROOT / args["cnn_train_dataset_path"]
    test_data_path = REPO_ROOT / args["cnn_test_dataset_path"]

    train_dataset = load_dataset(train_data_path)
    test_dataset = load_dataset(test_data_path)

    # 确保序列长度至少为最大卷积核大小
    max_kernel_size = max(args.get("kernel_sizes", [3, 4, 5]))
    train_dataset = ensure_min_length(train_dataset, min_length=max_kernel_size)
    test_dataset = ensure_min_length(test_dataset, min_length=max_kernel_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=args.get("num_workers", 0),
        collate_fn=collate_fn,
        pin_memory=True if device.type == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args.get("num_workers", 0),
        collate_fn=collate_fn,
        pin_memory=True if device.type == "cuda" else False,
    )

    # ----------------------------
    # 模型定义
    # ----------------------------
    EMBED_DIM = 512
    model = TextCNN(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_classes=len(label2idx),
        kernel_sizes=args.get("kernel_sizes", [3, 4, 5]),
        dropout=args.get("dropout", 0.5),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])

    # ----------------------------
    # 训练参数
    # ----------------------------
    epochs = args["epochs"]
    best_acc = 0.0
    patience = args.get("patience", 5)
    patience_counter = 0
    model_save_dir = Path(REPO_ROOT / args["cnn_model_path"])
    model_save_dir.mkdir(exist_ok=True)

    final_model_path = model_save_dir / "textcnn_final.pth"
    best_model_path = model_save_dir / "textcnn_best.pth"

    logger.info("开始训练...")
    for epoch in range(epochs):
        # ----------------------------
        # 训练阶段
        # ----------------------------
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for data, labels in progress_bar:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_loader)

        # ----------------------------
        # 验证阶段
        # ----------------------------
        val_loss, val_acc = evaluate(model, test_loader, device, criterion)

        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # ----------------------------
        # 模型保存 & 早停判断
        # ----------------------------
        if val_acc > best_acc:
            best_acc = val_acc
            save_model(model, best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        # 早停
        if patience > 0 and patience_counter >= patience:
            logger.info(f"验证准确率连续 {patience} 轮未提升，触发早停。")
            break

    # 保存最终模型
    save_model(model, final_model_path)
    logger.info(f"训练结束。最佳验证准确率: {best_acc:.4f}")

    # ----------------------------
    # 推理示例
    # ----------------------------
    logger.info("开始推理示例...")

    # 初始化推理器
    classifier = TextClassifier(
        model_path=best_model_path,
        word2idx=word2idx,
        idx2label=idx2label,
        device=device,
    )

    # 示例文本（确保有足够的长度）
    test_texts = [
        "这个电影真的很精彩，演员表演出色，剧情扣人心弦",
        "产品质量很差，完全不值得购买，非常失望",
        "今天的天气不错，适合出门散步，阳光明媚",
    ]

    # 批量推理
    results = classifier.predict_batch(test_texts)

    # 输出结果
    logger.info("推理结果:")
    for i, (text, result) in enumerate(zip(test_texts, results)):
        logger.info(f"文本 {i+1}: {text}")
        logger.info(
            f"  预测: {result['prediction']}, 置信度: {result['confidence']:.4f}"
        )
        logger.info("---")


if __name__ == "__main__":
    args = parse_args()
    main(args)
