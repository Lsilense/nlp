import torch
import torch.nn as nn
from torchcrf import CRF
from typing import List, Dict
from transformers import AutoModel


class BERT_LSTM_CRF(nn.Module):
    def __init__(
        self, bert_model_name, tag_size, hidden_dim=200, lstm_layers=1, dropout=0.1
    ):
        super().__init__()

        # BERT 模型
        self.bert = AutoModel.from_pretrained(bert_model_name)
        bert_dim = self.bert.config.hidden_size

        # 冻结 BERT 参数（可选，根据需求决定是否微调）
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        # LSTM 层
        self.lstm = nn.LSTM(
            bert_dim,
            hidden_dim // 2,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Dropout 层
        self.dropout = nn.Dropout(dropout)

        # 输出层
        self.hidden2tag = nn.Linear(hidden_dim, tag_size)

        # CRF 层
        self.crf = CRF(tag_size, batch_first=True)

    def forward(self, input_ids, attention_mask, tags=None, token_type_ids=None):
        # BERT 输出
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        bert_embeddings = bert_outputs.last_hidden_state

        # 应用 dropout
        lstm_input = self.dropout(bert_embeddings)

        # LSTM 输出
        lstm_out, _ = self.lstm(lstm_input)
        lstm_out = self.dropout(lstm_out)

        # 发射分数
        emissions = self.hidden2tag(lstm_out)

        # 训练模式：计算 CRF 损失
        if tags is not None:
            # 注意：CRF 需要 mask 来处理 padding
            mask = attention_mask.bool()
            loss = -self.crf(emissions, tags, mask=mask, reduction="mean")
            return loss
        # 预测模式：返回发射分数
        else:
            return emissions

    def predict(
        self,
        sentence: List[str],  # ['北','京','是','中','国']
        tokenizer,
        label_map: Dict[str, int],
        device: str = "cpu",
    ):
        """对单句进行NER预测，修复索引对齐问题"""
        self.eval()
        device = torch.device(device)

        # 构建反转的标签映射
        label_map_rev = {v: k for k, v in label_map.items()}

        # 处理每个字符的tokenization
        all_tokens = ["[CLS]"]
        word_to_token_mapping = []  # 记录每个原始字符对应的token范围

        for word in sentence:
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) == 0:
                word_tokens = ["[UNK]"]

            # 记录这个原始字符对应的token在all_tokens中的起始位置
            start_idx = len(all_tokens)
            all_tokens.extend(word_tokens)
            end_idx = len(all_tokens)  # 结束位置（不包括）

            word_to_token_mapping.append((start_idx, end_idx))

        # 添加[SEP]
        all_tokens.append("[SEP]")

        # 转换为ID
        input_ids = tokenizer.convert_tokens_to_ids(all_tokens)
        attention_mask = [1] * len(input_ids)

        # Padding处理
        max_len = 64  # 必须和训练时一致
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            attention_mask = attention_mask[:max_len]
            # 需要相应地调整word_to_token_mapping
            for i in range(len(word_to_token_mapping)):
                start, end = word_to_token_mapping[i]
                if start >= max_len:
                    word_to_token_mapping[i] = (max_len - 1, max_len - 1)
                elif end > max_len:
                    word_to_token_mapping[i] = (start, max_len - 1)
        else:
            pad_length = max_len - len(input_ids)
            input_ids.extend([tokenizer.pad_token_id] * pad_length)
            attention_mask.extend([0] * pad_length)

        # 转为Tensor
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)

        # 前向传播
        with torch.no_grad():
            emissions = self(input_ids, attention_mask)
            mask = attention_mask.bool()
            predictions = self.crf.decode(emissions, mask=mask)[0]  # 解码第一个样本

        # 关键修复：正确处理[CLS]和子词对齐
        pred_labels = []

        for i, word in enumerate(sentence):
            start_idx, end_idx = word_to_token_mapping[i]

            # 确保索引在有效范围内
            if start_idx < len(predictions) and start_idx > 0:  # start_idx>0跳过[CLS]
                # 取该字符第一个子词的预测作为标签
                pred_label_idx = predictions[start_idx]
                pred_label = label_map_rev.get(pred_label_idx, "O")
            else:
                pred_label = "O"  # 回退标签

            pred_labels.append(pred_label)

        return list(zip(sentence, pred_labels))
