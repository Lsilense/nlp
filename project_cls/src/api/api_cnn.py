from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import torch
import uvicorn
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional
import time
import sys
from functools import cached_property
import json
import hashlib

# 设置项目根目录
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from src.utils.utils import init_logger
from src.models.cnn.model_cnn import TextClassifier
from config.config import parse_args

logger = init_logger()

# ==============================
# 请求/响应模型
# ==============================


class CacheRequest(BaseModel):
    text: str
    category: str
    confidence: float = 1.0


class CacheResponse(BaseModel):
    success: bool
    message: str
    cache_key: Optional[str] = None


class RemoveCacheRequest(BaseModel):
    text: str


class RemoveCacheResponse(BaseModel):
    success: bool
    message: str


class PredictRequest(BaseModel):
    text: str


class PredictionResult(BaseModel):
    prediction: str
    confidence: float
    all_probabilities: Dict[str, float]
    inference_time: float
    source: str  # cache/model


# ==============================
# 缓存管理器
# ==============================


class CacheManager:
    def __init__(self, cache_file: str = REPO_ROOT / "data/prediction_cache.json"):
        self.cache_file = Path(cache_file)
        self._cache = {}
        self.load_cache()

    def generate_cache_key(self, text: str) -> str:
        """为文本生成唯一的缓存键"""
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def load_cache(self):
        """从文件加载缓存"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.info(f"缓存加载成功: {len(self._cache)} 条记录")
            else:
                self._cache = {}
        except Exception as e:
            logger.error(f"缓存加载失败: {e}")
            self._cache = {}

    def save_cache(self):
        """保存缓存到文件"""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"缓存保存失败: {e}")

    def get_cached_result(self, text: str) -> Optional[Dict]:
        """获取缓存结果"""
        return self._cache.get(self.generate_cache_key(text))

    def add_cache(self, text: str, category: str, confidence: float = 1.0) -> bool:
        """添加缓存记录"""
        try:
            cache_key = self.generate_cache_key(text)
            self._cache[cache_key] = {
                "text": text,
                "prediction": category,
                "confidence": confidence,
                "all_probabilities": {category: confidence},
                "inference_time": 0.0,
                "cached_at": time.time(),
            }
            self.save_cache()
            logger.info(f"缓存添加: '{text[:50]}...' -> {category}")
            return True
        except Exception as e:
            logger.error(f"缓存添加失败: {e}")
            return False

    def remove_cache(self, text: str) -> bool:
        """移除缓存记录"""
        try:
            cache_key = self.generate_cache_key(text)
            if cache_key in self._cache:
                del self._cache[cache_key]
                self.save_cache()
                logger.info(f"缓存移除: '{text[:50]}...'")
                return True
            return False
        except Exception as e:
            logger.error(f"缓存移除失败: {e}")
            return False

    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        return {
            "total_entries": len(self._cache),
            "cache_file": str(self.cache_file),
            "cache_file_exists": self.cache_file.exists(),
        }


# ==============================
# 模型管理器
# ==============================


class ModelManager:
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self._model = None
        self._word2idx = None
        self._idx2label = None
        self._device = None

    @cached_property
    def args(self):
        return parse_args()

    def load_mapping(self, path: str) -> Dict[str, int]:
        """加载映射文件"""
        mapping = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, idx_str = line.split()
                    mapping[key] = int(idx_str)
        logger.info(f"映射加载: {path} -> {len(mapping)} 项")
        return mapping

    def initialize(self):
        """初始化模型"""
        if self._model is not None:
            return

        try:
            # 设置设备
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"使用设备: {self._device}")

            # 加载映射文件
            vocab_path = REPO_ROOT / self.args["cnn_vocab_path"]
            label_path = REPO_ROOT / self.args["cnn_label_path"]

            self._word2idx = self.load_mapping(str(vocab_path))
            label2idx = self.load_mapping(str(label_path))
            self._idx2label = {v: k for k, v in label2idx.items()}

            # 加载模型
            model_path = REPO_ROOT / self.args["cnn_model_path"] / "textcnn_best.pth"
            if not model_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {model_path}")

            self._model = TextClassifier(
                model_path=str(model_path),
                word2idx=self._word2idx,
                idx2label=self._idx2label,
                device=self._device,
            )

            logger.info("模型初始化完成")

        except Exception as e:
            logger.critical(f"模型初始化失败: {e}")
            raise

    def predict(self, text: str) -> Dict[str, Any]:
        """执行推理"""
        if self._model is None:
            raise RuntimeError("模型未初始化")

        # 检查缓存
        cached_result = self.cache_manager.get_cached_result(text)
        if cached_result:
            logger.info(f"缓存命中: '{text[:50]}...'")
            return {**cached_result, "source": "cache"}

        # 模型推理
        start_time = time.time()
        result = self._model.predict(text)
        result.update({"inference_time": time.time() - start_time, "source": "model"})

        logger.info(f"模型推理: '{text[:50]}...' -> {result['prediction']}")
        return result

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def vocab_size(self) -> int:
        return len(self._word2idx) if self._word2idx else 0

    @property
    def num_classes(self) -> int:
        return len(self._idx2label) if self._idx2label else 0

    @property
    def labels(self) -> List[str]:
        return list(self._idx2label.values()) if self._idx2label else []


# ==============================
# 全局管理器
# ==============================

cache_manager = CacheManager()
model_manager = ModelManager(cache_manager)

# ==============================
# 生命周期管理
# ==============================


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("服务启动中...")
    model_manager.initialize()
    logger.info("服务启动成功")
    yield
    logger.info("服务关闭中...")


# ==============================
# FastAPI 应用
# ==============================

app = FastAPI(
    title="文本分类API",
    description="基于TextCNN的文本分类服务",
    version="1.0.0",
    lifespan=lifespan,
)

# ==============================
# API 路由
# ==============================


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_ready": model_manager.is_ready,
        "device": str(model_manager._device) if model_manager.is_ready else None,
        "vocab_size": model_manager.vocab_size,
        "num_classes": model_manager.num_classes,
        "labels": model_manager.labels,
        "cache_stats": cache_manager.get_cache_stats(),
    }


@app.post("/predict", response_model=PredictionResult)
async def predict(request: PredictRequest):
    """文本分类推理"""
    if not model_manager.is_ready:
        raise HTTPException(503, "服务不可用")

    try:
        result = model_manager.predict(request.text)
        return PredictionResult(**result)
    except Exception as e:
        logger.error(f"推理失败: {e}")
        raise HTTPException(500, f"推理错误: {str(e)}")


@app.post("/cache/add", response_model=CacheResponse)
async def add_cache(request: CacheRequest):
    """添加缓存"""
    if not model_manager.is_ready:
        return CacheResponse(success=False, message="服务未就绪")

    if request.category not in model_manager.labels:
        return CacheResponse(
            success=False, message=f"无效类别，支持: {model_manager.labels}"
        )

    success = cache_manager.add_cache(
        text=request.text, category=request.category, confidence=request.confidence
    )

    if success:
        cache_key = cache_manager.generate_cache_key(request.text)
        return CacheResponse(
            success=True,
            message="缓存添加成功",
            cache_key=cache_key,
        )
    return CacheResponse(success=False, message="缓存添加失败")


@app.post("/cache/remove", response_model=RemoveCacheResponse)
async def remove_cache(request: RemoveCacheRequest):
    """移除缓存"""
    success = cache_manager.remove_cache(request.text)
    message = "缓存移除成功" if success else "缓存不存在"
    return RemoveCacheResponse(success=success, message=message)


if __name__ == "__main__":
    uvicorn.run(
        "api_cnn:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
