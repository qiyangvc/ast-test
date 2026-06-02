#!/usr/bin/env python
"""训练脚本入口"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.utils import setup_logging
from src.word2vec import train_word2vec
from src.text_features import prepare_mlp_features, prepare_sequence_features
from src.classifier import train_rnn_classifier, train_mlp_classifier, train_cnn_classifier


def main():
    setup_logging()
    Config.ensure_dirs()
    
    print("=" * 60)
    print("开始训练垃圾文本分类系统")
    print("=" * 60)
    
    # 1. 训练Word2Vec词向量
    print("\n[1/4] 训练Word2Vec词向量...")
    try:
        train_word2vec()
        print("[OK] Word2Vec训练完成")
    except Exception as e:
        print(f"[ERROR] Word2Vec训练失败: {e}")
        return
    
    # 2. 准备特征
    print("\n[2/4] 准备特征数据...")
    try:
        prepare_mlp_features()
        prepare_sequence_features()
        print("[OK] 特征数据准备完成")
    except Exception as e:
        print(f"[ERROR] 特征准备失败: {e}")
        return
    
    # 3. 训练分类器
    print("\n[3/4] 训练RNN分类器...")
    try:
        train_rnn_classifier()
        print("[OK] RNN分类器训练完成")
    except Exception as e:
        print(f"[ERROR] RNN训练失败: {e}")
    
    print("\n[4/4] 训练MLP分类器...")
    try:
        train_mlp_classifier()
        print("[OK] MLP分类器训练完成")
    except Exception as e:
        print(f"[ERROR] MLP训练失败: {e}")
    
    print("\n[5/5] 训练CNN分类器...")
    try:
        train_cnn_classifier()
        print("[OK] CNN分类器训练完成")
    except Exception as e:
        print(f"[ERROR] CNN训练失败: {e}")
    
    print("\n" + "=" * 60)
    print("训练流程全部完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
