# cls-classification-model

## 项目概述
该项目是一个分类模型的实现，旨在通过使用不同的深度学习模型（如BERT和CNN）对文本数据进行分类。项目提供了完整的训练、评估和API接口，方便用户进行模型的使用和部署。

## 目录结构
```
cls-classification-model
├── README.md          # 项目文档
├── config             # 配置文件
│   ├── config_local.yaml  # 本地配置文件
│   └── config.py      # 配置的Python实现
├── data               # 数据集
│   ├── test.txt       # 测试数据
│   └── train.txt      # 训练数据
├── scripts            # 启动脚本
│   └── run_start.sh   # 启动项目的Shell脚本
├── src                # 源代码
│   ├── api            # API接口
│   │   └── api.py     # API实现
│   ├── models         # 模型实现
│   │   ├── bert       # BERT模型
│   │   │   ├── data_prepare.py  # 数据准备
│   │   │   ├── main.py  # BERT模型主入口
│   │   │   └── model_bert.py  # BERT模型结构
│   │   └── cnn        # CNN模型
│   │       ├── data_prepare.py  # 数据准备
│   │       ├── main.py  # CNN模型主入口
│   │       └── model_cnn.py  # CNN模型结构
│   └── utils          # 工具函数
│       └── utils.py    # 通用工具函数
└── requirements.txt   # Python依赖包
```

## 使用方法
1. **环境配置**：请确保安装了项目所需的依赖包，可以通过以下命令安装：
   ```
   pip install -r requirements.txt
   ```

2. **数据准备**：将训练数据和测试数据放置在`data/`目录下，格式应符合模型要求。

3. **运行项目**：使用以下命令启动项目：
   ```
   bash scripts/run_start.sh
   ```

4. **API接口**：项目提供了API接口，可以通过`src/api/api.py`进行访问和交互。

## 贡献
欢迎任何形式的贡献！请提交问题或拉取请求以帮助改进项目。

## 许可证
该项目遵循MIT许可证，详细信息请查看LICENSE文件。