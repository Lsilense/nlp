from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import torch
import uvicorn
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
import time
import sys
from functools import cached_property
import json
import hashlib

# 设置项目根目录
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from src.utils.utils import init_logger
from src.models.lstm.model_lstm import LSTM_CRF
from config.config import parse_args

logger = init_logger()


# ==============================
# 请求/响应模型
# ==============================


class EntityInfo(BaseModel):
    """实体基本信息（用于缓存请求）"""

    text: str
    type: str


class Entity(BaseModel):
    """实体完整信息（包含位置）"""

    text: str
    type: str
    start: int
    end: int


class CacheRequest(BaseModel):
    text: str
    entities: List[EntityInfo]
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


class NERRequest(BaseModel):
    text: str


class NERResponse(BaseModel):
    """NER响应结果"""

    text: str
    entities: List[Entity]
    inference_time: float
    source: str  # cache/model


# ==============================
# 缓存管理器
# ==============================


class CacheManager:
    """管理NER结果的缓存"""

    def __init__(self, cache_file: str = REPO_ROOT / "data/ner_cache.json"):
        self.cache_file = Path(cache_file)
        self._cache = {}
        self._supported_entity_types = None
        self.load_cache()

    def set_supported_entity_types(self, entity_types: set):
        """设置支持的实体类型"""
        self._supported_entity_types = entity_types
        logger.info(f"设置支持的实体类型: {len(entity_types)} 种")

    def _validate_entity_types(self, entities: List[EntityInfo]) -> bool:
        """验证实体类型是否在支持的范围内"""
        if not self._supported_entity_types:
            logger.warning("尚未设置支持的实体类型，跳过类型检查")
            return True

        invalid_entities = [
            f"'{entity.text}': {entity.type}"
            for entity in entities
            if entity.type not in self._supported_entity_types
        ]

        if invalid_entities:
            logger.warning(f"发现不支持的实体类型: {', '.join(invalid_entities)}")
            return False

        return True

    def generate_cache_key(self, text: str) -> str:
        """为文本生成唯一的缓存键"""
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def _find_entity_position(
        self, text: str, entity_info: EntityInfo
    ) -> Optional[Dict]:
        """在文本中查找实体的位置"""
        start_pos = text.find(entity_info.text)
        if start_pos == -1:
            logger.warning(f"在文本中未找到实体 '{entity_info.text}'")
            return None

        return {
            "text": entity_info.text,
            "type": entity_info.type,
            "start": start_pos,
            "end": start_pos + len(entity_info.text),
        }

    def load_cache(self):
        """从文件加载缓存"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.info(f"NER缓存加载成功: {len(self._cache)} 条记录")
            else:
                self._cache = {}
        except Exception as e:
            logger.error(f"NER缓存加载失败: {e}")
            self._cache = {}

    def save_cache(self):
        """保存缓存到文件"""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"NER缓存保存失败: {e}")

    def get_cached_result(self, text: str) -> Optional[Dict]:
        """获取缓存结果"""
        return self._cache.get(self.generate_cache_key(text))

    def add_cache(
        self, text: str, entities: List[EntityInfo], confidence: float = 1.0
    ) -> bool:
        """添加缓存记录"""
        try:
            # 验证实体类型
            if not self._validate_entity_types(entities):
                logger.error("实体类型检查失败，拒绝添加缓存")
                return False

            # 查找实体位置
            entities_with_positions = []
            for entity_info in entities:
                entity_with_position = self._find_entity_position(text, entity_info)
                if entity_with_position:
                    entities_with_positions.append(entity_with_position)
                else:
                    logger.warning(
                        f"跳过实体 '{entity_info.text}'，在文本中未找到匹配位置"
                    )

            if not entities_with_positions:
                logger.warning("没有找到任何有效实体位置，跳过缓存")
                return False

            # 保存缓存
            cache_key = self.generate_cache_key(text)
            self._cache[cache_key] = {
                "text": text,
                "entities": entities_with_positions,
                "inference_time": 0.0,
                "cached_at": time.time(),
                "confidence": confidence,
            }
            self.save_cache()
            logger.info(
                f"NER缓存添加: '{text[:50]}...' -> {len(entities_with_positions)}个实体"
            )
            return True

        except Exception as e:
            logger.error(f"NER缓存添加失败: {e}")
            return False

    def remove_cache(self, text: str) -> bool:
        """移除缓存记录"""
        try:
            cache_key = self.generate_cache_key(text)
            if cache_key in self._cache:
                del self._cache[cache_key]
                self.save_cache()
                logger.info(f"NER缓存移除: '{text[:50]}...'")
                return True
            return False
        except Exception as e:
            logger.error(f"NER缓存移除失败: {e}")
            return False

    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        total_entities = sum(len(item["entities"]) for item in self._cache.values())
        return {
            "total_entries": len(self._cache),
            "total_entities": total_entities,
            "cache_file": str(self.cache_file),
            "cache_file_exists": self.cache_file.exists(),
        }


# ==============================
# 模型管理器
# ==============================


class ModelManager:
    """管理模型加载和推理"""

    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self._model = None
        self._vocab = None
        self._label_map = None
        self._device = None

    @cached_property
    def args(self):
        """解析并缓存配置参数"""
        return parse_args()

    def _load_mapping_file(self, file_path: str) -> Dict[str, int]:
        """加载映射文件（词汇表或标签映射）"""
        mapping = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                key, value = line.strip().split("\t")
                mapping[key] = int(value)
        return mapping

    @property
    def supported_entity_types(self) -> set:
        """获取支持的实体类型集合"""
        if not self._label_map:
            return set()

        # 从标签映射中提取实体类型（去掉B-/I-前缀）
        return {
            label[2:]  # 去掉前缀
            for label in self._label_map.keys()
            if label.startswith(("B-", "I-"))
        }

    def initialize(self):
        """初始化模型"""
        if self._model is not None:
            return

        try:
            # 设置设备
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"使用设备: {self._device}")

            # 加载词汇表和标签映射
            vocab_path = REPO_ROOT / self.args["lstm_vocab_path"]
            label_map_path = REPO_ROOT / self.args["lstm_label_path"]

            self._vocab = self._load_mapping_file(str(vocab_path))
            self._label_map = self._load_mapping_file(str(label_map_path))

            logger.info(f"加载词汇表: {len(self._vocab)} 个词")
            logger.info(f"加载标签映射: {len(self._label_map)} 个标签")

            # 设置缓存管理器支持的实体类型
            self.cache_manager.set_supported_entity_types(self.supported_entity_types)

            # 创建并加载模型
            self._model = LSTM_CRF(
                vocab_size=len(self._vocab),
                tag_size=len(self._label_map),
                embed_dim=self.args.get("embed_dim", 100),
                hidden_dim=self.args.get("hidden_dim", 200),
            )

            model_path = REPO_ROOT / self.args["lstm_model_path"] / "ner_lstm_best.pth"
            if not model_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {model_path}")

            state_dict = torch.load(
                model_path, map_location=self._device, weights_only=True
            )
            self._model.load_state_dict(state_dict)
            self._model = self._model.to(self._device)
            self._model.eval()

            logger.info(f"模型加载完成: {model_path}")

        except Exception as e:
            logger.critical(f"模型初始化失败: {e}", exc_info=True)
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
                entity_type = label[2:]
                current_entity = Entity(
                    text=token, type=entity_type, start=i, end=i + 1
                )

            elif label.startswith("I-") and current_entity:
                # 继续当前实体
                entity_type = label[2:]
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

    def _dict_to_entities(self, entities_dict: List[Dict]) -> List[Entity]:
        """将字典格式转换为Entity对象"""
        return [
            Entity(
                text=entity["text"],
                type=entity["type"],
                start=entity["start"],
                end=entity["end"],
            )
            for entity in entities_dict
        ]

    def predict(self, text: str) -> Dict[str, Any]:
        """执行推理并返回结构化的结果"""
        if self._model is None:
            raise RuntimeError("模型未初始化")

        try:
            start_time = time.time()

            # 检查缓存
            cached_result = self.cache_manager.get_cached_result(text)
            if cached_result:
                logger.info(f"NER缓存命中: '{text[:50]}...'")
                return {
                    "text": text,
                    "entities": self._dict_to_entities(cached_result["entities"]),
                    "inference_time": time.time() - start_time,
                    "source": "cache",
                }

            # 按字符切分并执行推理
            tokens = list(text.strip())
            raw_result = self._model.predict(
                tokens, self._vocab, self._label_map, device=self._device
            )

            # 转换结果为结构化格式
            tokens_result = [item[0] for item in raw_result]
            labels_result = [item[1] for item in raw_result]
            entities = self._convert_to_entities(tokens_result, labels_result)
            inference_time = time.time() - start_time

            logger.info(
                f"NER模型推理: '{text[:50]}...' -> {len(entities)}个实体, "
                f"耗时: {inference_time:.3f}s"
            )

            return {
                "text": text,
                "entities": entities,
                "inference_time": inference_time,
                "source": "model",
            }

        except Exception as e:
            logger.error(f"推理失败: {e}", exc_info=True)
            raise

    @property
    def is_ready(self) -> bool:
        """检查模型是否就绪"""
        return self._model is not None

    @property
    def vocab_size(self) -> int:
        return len(self._vocab) if self._vocab else 0

    @property
    def num_labels(self) -> int:
        return len(self._label_map) if self._label_map else 0

    @property
    def labels(self) -> List[str]:
        return list(self._label_map.keys()) if self._label_map else []


# ==============================
# 全局管理器和应用初始化
# ==============================

cache_manager = CacheManager()
model_manager = ModelManager(cache_manager)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务生命周期管理"""
    logger.info("正在启动NER服务...")

    try:
        model_manager.initialize()
        logger.info("NER服务启动成功！")
    except Exception as e:
        logger.critical(f"NER服务启动失败: {e}")
        raise RuntimeError(f"无法启动NER服务: {e}")

    yield  # 服务运行中

    logger.info("NER服务正在关闭...")


app = FastAPI(
    title="NER 推理服务",
    description="基于 LSTM-CRF 的命名实体识别 API",
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
        "vocab_size": model_manager.vocab_size,
        "num_labels": model_manager.num_labels,
        "labels": model_manager.labels,
        "supported_entity_types": list(model_manager.supported_entity_types),
        "cache_stats": cache_manager.get_cache_stats(),
    }


@app.post("/predict", response_model=NERResponse)
async def predict(request: NERRequest):
    """执行 NER 推理"""
    if not model_manager.is_ready:
        raise HTTPException(status_code=503, detail="模型未就绪，服务不可用")

    try:
        result = model_manager.predict(request.text)
        return NERResponse(**result)
    except Exception as e:
        logger.error(f"NER推理请求处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"NER推理服务错误: {str(e)}")


@app.post("/cache/add", response_model=CacheResponse)
async def add_cache(request: CacheRequest):
    """添加NER缓存"""
    if not model_manager.is_ready:
        return CacheResponse(success=False, message="NER服务未就绪")

    success = cache_manager.add_cache(
        text=request.text, entities=request.entities, confidence=request.confidence
    )

    if success:
        cache_key = cache_manager.generate_cache_key(request.text)
        return CacheResponse(
            success=True, message="NER缓存添加成功", cache_key=cache_key
        )

    return CacheResponse(
        success=False,
        message=f"NER缓存添加失败,支持的类别{model_manager.supported_entity_types}",
    )


@app.post("/cache/remove", response_model=RemoveCacheResponse)
async def remove_cache(request: RemoveCacheRequest):
    """移除NER缓存"""
    success = cache_manager.remove_cache(request.text)
    message = "NER缓存移除成功" if success else "NER缓存不存在"
    return RemoveCacheResponse(success=success, message=message)


if __name__ == "__main__":
    uvicorn.run(
        "api_lstm:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        workers=1,
    )
