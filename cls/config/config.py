import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def _apply_config(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """将配置字典值写入 args，仅更新已存在的参数，支持一层嵌套。"""
    for key, value in (config or {}).items():
        if isinstance(value, dict):
            for k, v in value.items():
                if hasattr(args, k):
                    setattr(args, k, v)
        elif hasattr(args, key):
            setattr(args, key, value)


def parse_args(argv: Optional[list] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="通用配置解析")

    # 基础配置
    parser.add_argument("--data_dir", type=str, default="data", help="数据目录")
    parser.add_argument("--model", type=str, default="cnn", help="模型类型")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader 的 num_workers")
    parser.add_argument("--max_len", type=int, default=64, help="最大序列长度")
    parser.add_argument("--bert_model", type=str, default="bert-base-chinese", help="BERT 模型名称")
    parser.add_argument("--num_classes", type=int, default=3, help="类别数")
    parser.add_argument("--output_dir", type=str, default="outputs", help="输出目录")
    parser.add_argument("--dropout", type=float, default=0.3, help="dropout 比例")
    parser.add_argument("--config_file", type=str, default="config_local.yaml", help="YAML 配置文件路径（可选）")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")

    # 数据文件
    parser.add_argument("--train_file", type=str, default="train.txt", help="训练文件路径")
    parser.add_argument("--test_file", type=str, default="test.txt", help="测试文件路径")

    args = parser.parse_args(argv)

    # 处理配置文件
    config_path = (Path(__file__).resolve().parent / args.config_file).resolve()
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            _apply_config(args, config)
        except Exception as e:
            raise RuntimeError(f"加载配置文件失败: {config_path} -> {e}")

    return vars(args)


if __name__ == "__main__":
    args = parse_args()
    print(args)
