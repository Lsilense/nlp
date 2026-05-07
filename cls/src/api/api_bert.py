from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import torch
import uvicorn
from contextlib import asynccontextmanager
from typing import Dict, List, Any
import time
import sys
from functools import cached_property

# 设置项目根目录
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from src.utils.utils import init_logger
from src.models.bert.model_bert import BertTextClassifierWrapper
from config.config import parse_args

logger = init_logger()


# ==============================
# 请求/响应模型
# ==============================

class PredictRequest(BaseModel):
    text: str


class PredictionResult(BaseModel):
    """预测结果"""
    prediction: str
    confidence: float
    all_probabilities: Dict[str, float]
    inference_time: float


# ==============================
# 模型管理器
# ==============================

class ModelManager:
    """管理BERT模型加载和推理"""

    def __init__(self):
        self._model = None
        self._idx2label = None
        self._device = None

    @cached_property
    def args(self):
        """解析并缓存配置参数"""
        return parse_args()

    def load_mapping(self, path: str) -> Dict[str, int]:
        """加载映射文件"""
        mapping = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"Invalid line: {line}")
                key, idx_str = parts
                mapping[key] = int(idx_str)
        logger.info(f"Loaded {len(mapping)} items from {path}")
        return mapping

    def initialize(self):
        """初始化BERT模型"""
        if self._model is not None:
            return

        try:
            # 1. 设置设备
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"使用设备: {self._device}")

            # 2. 加载标签映射
            label_path = REPO_ROOT / self.args["bert_label_path"]
            label2idx = self.load_mapping(str(label_path))
            self._idx2label = {v: k for k, v in label2idx.items()}

            # 3. 创建并加载BERT模型
            model_path = REPO_ROOT / self.args["bert_model_save_path"] / "bert_classifier_best.pth"
            pretrained_model_path = REPO_ROOT / self.args["bert_model_name_or_path"]

            if not model_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {model_path}")

            self._model = BertTextClassifierWrapper(
                model_path=str(model_path),
                idx2label=self._idx2label,
                pretrained_model_name=str(pretrained_model_path),
                device=self._device.type,
            )

            logger.info(f"BERT模型加载完成: {model_path}")

        except Exception as e:
            logger.critical(f"BERT模型初始化失败: {e}", exc_info=True)
            raise

    def predict(self, text: str) -> Dict[str, Any]:
        """执行BERT推理并返回结构化的结果"""
        if self._model is None:
            raise RuntimeError("模型未初始化")

        try:
            start_time = time.time()
            
            # 执行BERT推理
            result = self._model.predict(text)
            
            # 计算推理时间
            inference_time = time.time() - start_time
            
            # 添加推理时间到结果中
            result["inference_time"] = inference_time
            
            return result

        except Exception as e:
            logger.error(f"BERT推理失败: {e}", exc_info=True)
            raise

    @property
    def is_ready(self) -> bool:
        """检查模型是否就绪"""
        return self._model is not None

    @property
    def num_classes(self) -> int:
        """返回类别数量"""
        return len(self._idx2label) if self._idx2label else 0

    @property
    def labels(self) -> List[str]:
        """返回所有标签"""
        return list(self._idx2label.values()) if self._idx2label else []


# ==============================
# 全局模型管理器
# ==============================

model_manager = ModelManager()


# ==============================
# 生命周期管理
# ==============================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务生命周期管理"""
    logger.info("正在启动BERT文本分类服务...")

    try:
        model_manager.initialize()
        logger.info("BERT服务启动成功！")
    except Exception as e:
        logger.critical(f"BERT服务启动失败: {e}")
        raise RuntimeError(f"无法启动BERT服务: {e}")

    yield  # 服务运行中

    logger.info("BERT服务正在关闭...")


# ==============================
# FastAPI 应用
# ==============================

app = FastAPI(
    title="BERT文本分类API",
    description="基于BERT的文本分类服务",
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
        "num_classes": model_manager.num_classes,
        "labels": model_manager.labels,
    }


@app.post("/predict", response_model=PredictionResult)
async def predict(request: PredictRequest):
    """执行BERT文本分类推理"""
    if not model_manager.is_ready:
        raise HTTPException(status_code=503, detail="BERT模型未就绪，服务不可用")

    try:
        result = model_manager.predict(request.text)

        logger.info(
            f"BERT推理完成 - 文本长度: {len(request.text)}, "
            f"预测类别: {result['prediction']}, "
            f"置信度: {result['confidence']:.4f}, "
            f"耗时: {result['inference_time']:.3f}s"
        )

        return PredictionResult(**result)

    except Exception as e:
        logger.error(f"BERT推理请求处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"BERT推理服务错误: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "api_bert:app",  # 注意：这里需要根据实际文件名调整
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        workers=1,
    )