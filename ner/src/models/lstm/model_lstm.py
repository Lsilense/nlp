import torch
import torch.nn as nn
from torchcrf import CRF
from typing import List, Dict


class LSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_size, embed_dim=100, hidden_dim=200):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim // 2, bidirectional=True, batch_first=True
        )
        self.hidden2tag = nn.Linear(hidden_dim, tag_size)
        self.crf = CRF(tag_size, batch_first=True)

    def forward(self, sentences, tags=None, mask=None):
        embeddings = self.embedding(sentences)
        lstm_out, _ = self.lstm(embeddings)
        emissions = self.hidden2tag(lstm_out)

        if tags is not None:
            return -self.crf(emissions, tags, mask=mask, reduction="mean")
        else:
            return emissions

    def predict(
        self,
        sentence: List[str],
        vocab: Dict[str, int],
        label_map: Dict[str, int],
        device: str = "cpu",
    ):
        """对单句进行NER预测（已整合到类中）"""
        self.eval()
        device = torch.device(device)

        # 转换为ID序列 (添加batch维度)
        word_ids = [vocab.get(word, vocab.get("[UNK]", 0)) for word in sentence]
        word_ids = torch.tensor([word_ids], dtype=torch.long).to(device)

        # 预测（自动调用forward获取emissions）
        with torch.no_grad():
            emissions = self(word_ids)  # 等价于 self.forward(word_ids)
            # 关键修复：确保emissions在指定设备上
            emissions = emissions.to(device)
            # CRF解码（batch_size=1时取第一个元素）
            predictions = self.crf.decode(emissions)[0]

        # 转换为标签
        label_map_rev = {v: k for k, v in label_map.items()}
        # 确保返回结果与输入长度一致
        assert len(predictions) == len(
            sentence
        ), f"预测长度 {len(predictions)} 与句子长度 {len(sentence)} 不匹配"

        return [
            (word, label_map_rev[pred]) for word, pred in zip(sentence, predictions)
        ]
