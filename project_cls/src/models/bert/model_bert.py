from pathlib import Path
import sys
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig

# 添加项目根目录
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))
from src.utils.utils import init_logger

logger = init_logger()


class BertTextClassifier(nn.Module):
    def __init__(
        self,
        pretrained_model_name="bert-base-chinese",
        num_classes=2,
        dropout=0.3,
        freeze_bert=False,  # 是否冻结BERT参数
    ):
        super(BertTextClassifier, self).__init__()
        
        # 加载BERT模型和配置
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.config = BertConfig.from_pretrained(pretrained_model_name)
        self.hidden_size = self.config.hidden_size
        
        # 分类层
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        
        # 可选：冻结BERT参数
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # BERT前向传播
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # 使用[CLS] token的隐藏状态进行分类
        pooled_output = outputs.pooler_output  # 或者使用 outputs.last_hidden_state[:, 0, :]
        
        # 分类
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits


class BertTextClassifierWrapper:
    """基于BERT的文本分类推理器"""

    def __init__(self, model_path, idx2label, pretrained_model_name="bert-base-chinese", device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.idx2label = idx2label
        self.model_path = model_path
        self.pretrained_model_name = pretrained_model_name

        # 初始化tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        
        # 初始化模型
        self.model = BertTextClassifier(
            pretrained_model_name=pretrained_model_name,
            num_classes=len(idx2label),
            dropout=0.3,
            freeze_bert=False
        ).to(self.device)

        # 加载模型权重
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()
        logger.info(f"BERT推理器初始化完成，使用设备: {self.device}")

    def preprocess_text(self, text, max_length=512):
        """使用BERT tokenizer预处理文本"""
        # 对文本进行编码
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # 添加[CLS]和[SEP]
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device),
            'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).to(self.device)
        }

    def predict(self, text):
        """预测单条文本"""
        with torch.no_grad():
            # 预处理
            inputs = self.preprocess_text(text)
            
            # 模型预测
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                token_type_ids=inputs['token_type_ids']
            )
            
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

    def predict_batch_optimized(self, texts, batch_size=32):
        """优化版的批量预测，减少内存使用"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # 批量编码
            encodings = self.tokenizer.batch_encode_plus(
                batch_texts,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            inputs = {
                'input_ids': encodings['input_ids'].to(self.device),
                'attention_mask': encodings['attention_mask'].to(self.device),
                'token_type_ids': encodings.get('token_type_ids', torch.zeros_like(encodings['input_ids'])).to(self.device)
            }
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs, dim=1)
                pred_indices = outputs.argmax(dim=1).cpu().numpy()
                
                for j, pred_idx in enumerate(pred_indices):
                    confidence = probs[j][pred_idx].item()
                    results.append({
                        "prediction": self.idx2label[pred_idx],
                        "confidence": confidence,
                        "all_probabilities": {
                            self.idx2label[i]: prob.item() for i, prob in enumerate(probs[j])
                        },
                    })
        
        return results