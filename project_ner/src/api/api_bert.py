from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import torch
import uvicorn
from contextlib import asynccontextmanager
from typing import List, Dict, Any
import time
import sys
from functools import cached_property

# 设置项目根目录
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from src.utils.utils import init_logger
from src.models.bert.model_bert import BERT_LSTM_CRF
from transformers import BertTokenizer
from config.config import parse_args

logger = init_logger()


# ==============================
# 请求/响应模型
# ==============================


class NERRequest(BaseModel):
    text: str


class Entity(BaseModel):
    """实体信息"""

    text: str
    type: str
    start: int
    end: int


class NERResponse(BaseModel):
    """NER响应结果"""

    text: str
    entities: List[Entity]
    inference_time: float


# ==============================
# 模型管理器
# ==============================


class BERTModelManager:
    """管理BERT模型加载和推理"""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._label_map = None
        self._device = None

    @cached_property
    def args(self):
        """解析并缓存配置参数"""
        return parse_args()

    def load_label_map(self, label_map_path: str) -> Dict[str, int]:
        """加载标签映射"""
        label_map = {}
        with open(label_map_path, "r", encoding="utf-8") as f:
            for line in f:
                label, idx = line.strip().split("\t")
                label_map[label] = int(idx)
        logger.info(f"加载标签映射: {len(label_map)} 个标签")
        return label_map

    def initialize(self):
        """初始化BERT模型和tokenizer"""
        if self._model is not None:
            return

        try:
            # 1. 设置设备
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"使用设备: {self._device}")

            # 2. 加载标签映射
            label_map_path = REPO_ROOT / self.args["bert_label_path"]
            self._label_map = self.load_label_map(str(label_map_path))

            # 3. 加载BERT tokenizer
            bert_model_path = REPO_ROOT / self.args["bert_model_name_or_path"]
            self._tokenizer = BertTokenizer.from_pretrained(str(bert_model_path))
            logger.info(f"加载BERT tokenizer: {bert_model_path}")

            # 4. 创建并加载BERT-CRF模型
            self._model = BERT_LSTM_CRF(
                bert_model_name=str(bert_model_path),
                tag_size=len(self._label_map),
                hidden_dim=self.args.get("hidden_dim", 200),
                lstm_layers=self.args.get("lstm_layers", 1),
                dropout=self.args.get("dropout", 0.1),
            )

            # 5. 加载训练好的模型权重
            model_path = (
                REPO_ROOT / self.args["bert_model_save_path"] / "ner_bert_best.pth"
            )
            if not model_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {model_path}")

            state_dict = torch.load(
                model_path, map_location=self._device, weights_only=True
            )
            self._model.load_state_dict(state_dict)
            self._model = self._model.to(self._device)
            self._model.eval()

            logger.info(f"BERT模型加载完成: {model_path}")

        except Exception as e:
            logger.critical(f"BERT模型初始化失败: {e}", exc_info=True)
            raise

    def _convert_to_entities(
        self, tokens: List[str], labels: List[str]
    ) -> List[Entity]:
        """将token-label对转换为实体列表"""
        entities = []
        current_entity = None

        for i, (token, label) in enumerate(zip(tokens, labels)):
            if label.startswith("B-"):
                # 开始新实体
                if current_entity:
                    entities.append(current_entity)
                entity_type = label[2:]  # 去掉B-前缀
                current_entity = Entity(
                    text=token, type=entity_type, start=i, end=i + 1
                )
            elif label.startswith("I-") and current_entity:
                # 继续当前实体
                entity_type = label[2:]  # 去掉I-前缀
                if entity_type == current_entity.type:
                    current_entity.text += token
                    current_entity.end = i + 1
                else:
                    # 类型不匹配，结束当前实体并开始新实体
                    entities.append(current_entity)
                    current_entity = Entity(
                        text=token, type=entity_type, start=i, end=i + 1
                    )
            else:
                # O标签或其他，结束当前实体
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        # 添加最后一个实体
        if current_entity:
            entities.append(current_entity)

        return entities

    def predict(self, text: str) -> Dict[str, Any]:
        """执行BERT模型推理并返回结构化的结果"""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("模型或tokenizer未初始化")

        try:
            start_time = time.time()

            # 1. 按字符切分输入句子
            tokens = list(text.strip())

            # 2. 执行BERT模型推理
            raw_result = self._model.predict(
                tokens, self._tokenizer, self._label_map, device=self._device
            )

            # 3. 转换结果为结构化格式
            tokens_result = [item[0] for item in raw_result]
            labels_result = [item[1] for item in raw_result]
            entities = self._convert_to_entities(tokens_result, labels_result)

            # 4. 计算推理时间
            inference_time = time.time() - start_time

            return {
                "text": text,
                "entities": entities,
                "inference_time": inference_time,
            }

        except Exception as e:
            logger.error(f"BERT推理失败: {e}", exc_info=True)
            raise

    @property
    def is_ready(self) -> bool:
        """检查模型是否就绪"""
        return self._model is not None and self._tokenizer is not None


# ==============================
# 全局模型管理器
# ==============================

model_manager = BERTModelManager()


# ==============================
# 生命周期管理
# ==============================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务生命周期管理"""
    logger.info("正在启动BERT NER服务...")

    try:
        model_manager.initialize()
        logger.info("BERT NER服务启动成功！")
    except Exception as e:
        logger.critical(f"BERT NER服务启动失败: {e}")
        raise RuntimeError(f"无法启动BERT NER服务: {e}")

    yield  # 服务运行中

    logger.info("BERT NER服务正在关闭...")


# ==============================
# FastAPI 应用
# ==============================

app = FastAPI(
    title="BERT NER 推理服务",
    description="基于 BERT-LSTM-CRF 的命名实体识别 API",
    version="1.0.0",
    lifespan=lifespan,
)


# ==============================
# API 路由
# ==============================


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "model_ready": model_manager.is_ready,
        "device": str(model_manager._device) if model_manager.is_ready else None,
        "model_type": "BERT-LSTM-CRF",
        "label_count": len(model_manager._label_map),
        "labels": list(model_manager._label_map.keys()),
        "bert_model": str(
            REPO_ROOT / model_manager.args["bert_model_save_path"] / "ner_bert_best.pth"
        ),
    }


@app.post("/predict", response_model=NERResponse)
async def predict(request: NERRequest):
    """执行 NER 推理"""
    if not model_manager.is_ready:
        raise HTTPException(status_code=503, detail="模型未就绪，服务不可用")

    try:
        result = model_manager.predict(request.text)

        logger.info(
            f"BERT推理完成 - 句子长度: {len(request.text)}, "
            f"实体数量: {len(result['entities'])}, "
            f"耗时: {result['inference_time']:.3f}s"
        )

        return NERResponse(**result)

    except Exception as e:
        logger.error(f"BERT推理请求处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"BERT推理服务错误: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "api_bert:app",  # 假设文件保存为api_bert.py
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        workers=1,
    )
