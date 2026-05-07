import json
from pathlib import Path
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

# 添加项目根目录
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

from config.config import parse_args
from src.utils.utils import init_logger
from src.models.bert.model_bert import BertTextClassifier, BertTextClassifierWrapper

logger = init_logger()


class TextDataset(Dataset):
    def __init__(self, data_dict):
        self.input_ids = data_dict["input_ids"]
        self.attention_mask = data_dict["attention_mask"]
        self.token_type_ids = data_dict["token_type_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "token_type_ids": torch.tensor(self.token_type_ids[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_data(path):
    """加载预处理好的 BERT 数据集（只包含 input_ids, attention_mask, label_id）"""
    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # 直接提取 tensor 数据
                input_ids_list.append(data["input_ids"])
                attention_mask_list.append(data["attention_mask"])
                token_type_ids_list.append(
                    data.get("token_type_ids", [0] * len(data["input_ids"]))
                )  # 兼容旧数据
                labels.append(int(data["label_id"]))
            except Exception as e:
                logger.error(f"解析错误 行{line_num}: {e}")
                continue

    logger.info(f"从 {path} 加载 {len(labels)} 条数据")
    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "token_type_ids": token_type_ids_list,
        "labels": labels,
    }


def load_labels(path):
    """加载标签映射"""
    label2idx, idx2label = {}, {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                label, idx = line.split()
                idx = int(idx)
                label2idx[label] = idx
                idx2label[idx] = label

    logger.info(f"加载 {len(label2idx)} 个标签")
    return label2idx, idx2label


def evaluate(model, data_loader, device, criterion):
    """评估模型"""
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for batch in data_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)

            outputs = model(**inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(data_loader), correct / total


def create_optimizer(model, args):
    """创建优化器"""
    no_decay = ["bias", "LayerNorm.weight"]
    grouped_params = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.get("weight_decay", 0.01),
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    return AdamW(grouped_params, lr=args["lr"], eps=args.get("adam_epsilon", 1e-8))


def train_epoch(model, train_loader, optimizer, scheduler, criterion, device, args):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="训练", leave=False)

    for batch in progress_bar:
        # 准备数据
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.get("max_grad_norm", 1.0)
        )
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(train_loader)


def main(args):
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 加载标签和模型
    label2idx, idx2label = load_labels(REPO_ROOT / args["bert_label_path"])

    model = BertTextClassifier(
        pretrained_model_name=REPO_ROOT / args["bert_model_name_or_path"],
        num_classes=len(label2idx),
        dropout=args.get("dropout", 0.3),
        freeze_bert=args.get("freeze_bert", False),
    ).to(device)

    # 准备数据
    train_data = load_data(REPO_ROOT / args["bert_train_dataset_path"])
    test_data = load_data(REPO_ROOT / args["bert_test_dataset_path"])

    train_dataset = TextDataset(train_data)
    test_dataset = TextDataset(test_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=args.get("num_workers", 0),
        pin_memory=device.type == "cuda",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args.get("num_workers", 0),
        pin_memory=device.type == "cuda",
    )

    # 训练准备
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, args)

    total_steps = len(train_loader) * args["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.get("warmup_steps", 0),
        num_training_steps=total_steps,
    )

    # 训练循环
    best_acc, patience_counter = 0.0, 0
    model_save_dir = Path(REPO_ROOT / args["bert_model_save_path"])
    model_save_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = model_save_dir / "bert_classifier_best.pth"
    final_model_path = model_save_dir / "bert_classifier_final.pth"

    logger.info("开始训练...")
    for epoch in range(args["epochs"]):
        # 训练
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, args
        )

        # 验证
        val_loss, val_acc = evaluate(model, test_loader, device, criterion)

        logger.info(
            f"Epoch {epoch+1}: 训练损失 {train_loss:.4f} | 验证损失 {val_loss:.4f} | 验证准确率 {val_acc:.4f}"
        )

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            logger.info(f"新的最佳准确率: {best_acc:.4f}")
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= args.get("patience", 5):
            logger.info("早停触发")
            break

    # 保存最终模型并测试
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"训练完成，最佳准确率: {best_acc:.4f}")

    # 推理示例
    logger.info("推理测试...")
    classifier = BertTextClassifierWrapper(
        model_path=best_model_path,
        idx2label=idx2label,
        pretrained_model_name=REPO_ROOT / args["bert_model_name_or_path"],
        device=device,
    )

    test_samples = [
        "这个电影真的很精彩，演员表演出色，剧情扣人心弦",
        "产品质量很差，完全不值得购买，非常失望",
        "今天的天气不错，适合出门散步，阳光明媚",
    ]

    results = classifier.predict_batch(test_samples)
    for i, (text, result) in enumerate(zip(test_samples, results)):
        logger.info(
            f"样本{i+1}: {text[:30]}... -> {result['prediction']} (置信度: {result['confidence']:.3f})"
        )


if __name__ == "__main__":
    main(parse_args())
