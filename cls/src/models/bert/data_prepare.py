from pathlib import Path
import sys
# 获取当前文件所在目录的上三级目录
sys.path.append(str(Path(__file__).resolve().parents[3]))
from config.config import parse_args


def main():
    """程序入口：解析参数并打印"""
    args = parse_args()
    print(args)


if __name__ == "__main__":
    main()
