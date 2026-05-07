from pathlib import Path
import sys
import jieba
import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加项目根目录
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))
from src.utils.utils import init_logger

logger = init_logger()


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_classes,
        kernel_sizes=[3, 4, 5],  # 增加卷积核尺寸范围
        num_filters=256,  # 增加卷积通道数
        dropout=0.5,
        use_batchnorm=True,  # 添加BatchNorm
    ):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=embed_dim, out_channels=num_filters, kernel_size=k
                    ),
                    nn.BatchNorm1d(num_filters) if use_batchnorm else nn.Identity(),
                    nn.ReLU(),
                )
                for k in kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x)  # (B, L, D)
        x = x.transpose(1, 2)  # (B, D, L)

        conv_outputs = []
        for conv in self.convs:
            c = conv(x)  # (B, num_filters, L - k + 1)
            pooled = F.max_pool1d(c, c.size(2))  # (B, num_filters, 1)
            conv_outputs.append(pooled.squeeze(2))  # (B, num_filters)

        x = torch.cat(conv_outputs, dim=1)  # (B, num_filters * num_kernels)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


class TextClassifier:
    """文本分类推理器"""

    def __init__(self, model_path, word2idx, idx2label, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.word2idx = word2idx
        self.idx2label = idx2label
        self.model_path = model_path

        # 使用优化后的模型参数
        self.model = TextCNN(
            vocab_size=len(word2idx),
            embed_dim=512,
            num_classes=len(idx2label),
            kernel_sizes=[3, 4, 5],  # 优化点1: 增加卷积核尺寸
            num_filters=256,  # 优化点2: 增加卷积通道数
            dropout=0.3,
            use_batchnorm=True,  # 优化点3: 添加BatchNorm
        ).to(self.device)

        # 加载模型权重
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()
        logger.info(f"推理器初始化完成，使用设备: {self.device}")

    def preprocess_text(self, text, min_length=5):
        words = list(jieba.cut(text))  # ← 替换这里
        input_ids = [self.word2idx.get(w, self.word2idx.get("[UNK]", 0)) for w in words]

        # 保证长度
        if len(input_ids) < min_length:
            input_ids += [self.word2idx.get("[PAD]", 0)] * (min_length - len(input_ids))

        return torch.tensor([input_ids], dtype=torch.long).to(self.device)

    def predict(self, text):
        """预测单条文本"""
        with torch.no_grad():
            input_tensor = self.preprocess_text(text)
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = outputs.argmax(dim=1).item()
            confidence = probs[0][pred_idx].item()

            return {
                "prediction": self.idx2label[pred_idx],
                "confidence": confidence,
                "all_probabilities": {
                    self.idx2label[i]: prob.item() for i, prob in enumerate(probs[0])
                },
            }

    def predict_batch(self, texts):
        """批量预测文本"""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results
