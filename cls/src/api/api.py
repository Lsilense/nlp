from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.models.bert.model_bert import ModelBERT
from src.models.cnn.model_cnn import CNNModel

app = FastAPI()


# 定义请求体
class TextRequest(BaseModel):
    text: str


# 初始化模型（此处需根据实际情况加载模型参数和配置）
bert_model = ModelBERT(...)  # 填写必要的配置或加载已训练权重
cnn_model = CNNModel(...)  # 填写必要的配置或加载已训练权重


@app.post("/predict/bert")
async def predict_bert(request: TextRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")
    prediction = bert_model.predict(request.text)
    return {"prediction": prediction}


@app.post("/predict/cnn")
async def predict_cnn(request: TextRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")
    prediction = cnn_model.predict(request.text)
    return {"prediction": prediction}
