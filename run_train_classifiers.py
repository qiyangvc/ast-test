"""训练分类器脚本"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.utils import setup_logging
from src.classifier import train_rnn_classifier, train_mlp_classifier, train_cnn_classifier

def main():
    setup_logging()
    Config.ensure_dirs()

    print("=" * 60)
    print("开始训练分类器")
    print("=" * 60)

    print("\n[1/3] 训练RNN分类器...")
    try:
        train_rnn_classifier()
        print("[OK] RNN分类器训练完成")
    except Exception as e:
        print(f"[ERROR] RNN训练失败: {e}")

    print("\n[2/3] 训练MLP分类器...")
    try:
        train_mlp_classifier()
        print("[OK] MLP分类器训练完成")
    except Exception as e:
        print(f"[ERROR] MLP训练失败: {e}")

    print("\n[3/3] 训练CNN分类器...")
    try:
        train_cnn_classifier()
        print("[OK] CNN分类器训练完成")
    except Exception as e:
        print(f"[ERROR] CNN训练失败: {e}")

    print("\n" + "=" * 60)
    print("分类器训练完成！")
    print("=" * 60)

if __name__ == '__main__':
    main()