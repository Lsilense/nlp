# 使用PyTorch官方GPU镜像（CUDA 11.8 + PyTorch 2.0.1）
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件（仅包含必要文件，避免大文件污染镜像）
COPY ./requirements.txt ./
COPY ./src ./src
COPY ./config ./config
COPY ./data ./data 

# 包含vocab.txt, labels.txt, model/textcnn_best.pth

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 设置环境变量（确保模型路径正确）
ENV CNN_VOCAB_PATH="data/vocab.txt" \
    CNN_LABEL_PATH="data/labels.txt" \
    CNN_MODEL_PATH="data/model"

# 暴露端口
EXPOSE 8000

# 启动命令（使用uvicorn运行）
CMD ["uvicorn", "src.api_cnn:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]