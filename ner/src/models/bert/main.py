import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Dict, Any
from pathlib import Path
from tqdm import tqdm
import sys
import os
from transformers import BertTokenizer

# 添加项目根目录
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

from config.config import parse_args
from src.utils.utils import init_logger
from src.models.bert.model_bert import BERT_LSTM_CRF

logger = init_logger()


class NERDataset(Dataset):
    """NER数据集，处理序列标注任务（BERT版本）"""

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
        padding_value=0,  # BERT的pad token id是0
    )

    padded_tags = pad_sequence(
        [torch.tensor(t, dtype=torch.long) for t in tags],
        batch_first=True,
        padding_value=0,  # 标签的填充值，对应O标签或其他
    )

    # 创建mask (1=有效, 0=填充)，确保第一个token总是有效的
    mask = (padded_words != 0).long()

    return padded_words, padded_tags, mask


def load_label_map(label_map_path: str) -> Dict[str, int]:
    """加载标签映射"""
    label_map = {}
    with open(label_map_path, "r", encoding="utf-8") as f:
        for line in f:
            label, idx = line.strip().split("\t")
            label_map[label] = int(idx)
    logger.info(f"加载标签映射: {len(label_map)} 个标签")
    return label_map


def load_dataset(
    dataset_path: str, tokenizer: BertTokenizer
) -> List[Tuple[List[int], List[int]]]:
    """加载NER数据集（BERT版本：直接使用预处理好的数据）"""
    dataset = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                words_str, labels_str = line.split("\t")
                # 直接使用预处理好的token IDs和label IDs
                token_ids = [int(token) for token in words_str.split()]
                tag_ids = [int(label) for label in labels_str.split()]

                dataset.append((token_ids, tag_ids))
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
            emissions = model(words, attention_mask=mask)
            predictions = model.crf.decode(emissions, mask=mask.bool())

            # 处理每个样本的预测
            for i in range(len(predictions)):
                # 获取当前样本的有效长度
                sample_mask = mask[i].bool()
                sample_len = sample_mask.sum().item()

                # 只比较有效部分的预测
                pred_tensor = torch.tensor(
                    predictions[i][:sample_len], dtype=torch.long, device=device
                )
                true_tags = tags[i][:sample_len]

                correct += (pred_tensor == true_tags).sum().item()
                total += sample_len

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
    model_save_path: str,
    bert_model_name: str = "bert-base-chinese",
) -> None:
    """训练BERT-CRF模型"""
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

    logger.info(f"开始训练BERT-CRF模型 (设备: {device}, BERT模型: {bert_model_name})")

    # 初始化最佳准确率
    best_acc = 0.0
    model_save_path = Path(REPO_ROOT / model_save_path)
    os.makedirs(model_save_path, exist_ok=True)
    final_model_path = model_save_path / "ner_bert_final.pth"
    best_model_path = model_save_path / "ner_bert_best.pth"

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False
        )

        for words, tags, mask in progress_bar:
            words, tags, mask = words.to(device), tags.to(device), mask.to(device)

            optimizer.zero_grad()
            loss = model(words, tags=tags, attention_mask=mask)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # 评估阶段
        train_loss = total_loss / len(train_loader)
        test_loss, test_acc = evaluate(model, test_loader, device)

        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.4f}"
        )

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"最佳模型已保存到: {best_model_path} (准确率: {test_acc:.4f})")

    # 保存最终模型
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"最后一轮模型已保存到: {final_model_path}")
    logger.info(f"训练完成，最佳准确率: {best_acc:.4f}")


def main(args: Dict[str, Any]):
    """主函数，执行训练和预测流程"""
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 初始化BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        REPO_ROOT / args["bert_model_name_or_path"]
    )
    logger.info(f"加载BERT tokenizer: {REPO_ROOT/args['bert_model_name_or_path']}")

    # 加载标签映射
    label_map = load_label_map(REPO_ROOT / args["bert_label_path"])

    # 加载数据集
    train_data = load_dataset(
        REPO_ROOT / args["bert_train_dataset_path"], tokenizer=tokenizer
    )
    test_data = load_dataset(
        REPO_ROOT / args["bert_test_dataset_path"], tokenizer=tokenizer
    )

    # 创建数据集
    train_dataset = NERDataset(train_data)
    test_dataset = NERDataset(test_data)

    # 创建BERT-CRF模型
    model = BERT_LSTM_CRF(
        bert_model_name=str(
            REPO_ROOT / args["bert_model_name_or_path"]
        ),  # 转换为字符串
        tag_size=len(label_map),
    )

    # # 训练模型
    # train_model(
    #     model,
    #     train_dataset,
    #     test_dataset,
    #     epochs=args["epochs"],
    #     batch_size=args["batch_size"],
    #     lr=args["lr_bert"],
    #     device=device,
    #     model_save_path=args[
    #         "bert_model_save_path"
    #     ],  # 不需要REPO_ROOT，train_model内部处理
    #     bert_model_name=str(REPO_ROOT / args["bert_model_name_or_path"]),
    # )

    # ====== 推理示例 ======
    # 加载最佳模型进行推理
    best_model = BERT_LSTM_CRF(
        bert_model_name=str(REPO_ROOT / args["bert_model_name_or_path"]),
        tag_size=len(label_map),
    )
    best_model.load_state_dict(
        torch.load(
            REPO_ROOT / args["bert_model_save_path"] / "ner_bert_best.pth",
            map_location=device,
            weights_only=True,
        )
    )
    best_model = best_model.to(device)
    best_model.eval()

    # 测试句子
    test_sentence = "今天北京的房价怎么样"
    tokens = list(test_sentence.strip())  # -> ['北', '京', '是', '中', '国', ...]

    print("\n=== 推理结果 ===")
    print(f"输入句子: {test_sentence}")
    print(f"分词结果: {' '.join(tokens)}")
    result = best_model.predict(tokens, tokenizer, label_map, device=device)
    print("预测结果:")
    for word, tag in result:
        print(f"{word} -> {tag}")
    print("================")


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    # 执行主流程
    main(args)
