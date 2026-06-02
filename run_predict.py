#!/usr/bin/env python
"""预测脚本入口"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.utils import setup_logging
from src.serving import TextClassifierService


def main():
    setup_logging()
    Config.ensure_dirs()
    
    # 创建服务
    service = TextClassifierService(model_type='rnn')
    
    print("=" * 60)
    print("垃圾文本分类预测服务")
    print("=" * 60)
    print("输入文本进行预测（输入 'exit' 退出）")
    print("-" * 60)
    
    while True:
        text = input("请输入文本: ")
        
        if text.lower() == 'exit':
            print("退出服务")
            break
        
        if not text.strip():
            print("请输入有效文本")
            continue
        
        try:
            result = service.predict(text)
            print(f"\n预测结果: {result['label']}")
            print(f"置信度: {result['confidence']:.4f}")
            print(f"使用模型: {result['model_type']}")
        except Exception as e:
            print(f"预测失败: {e}")
        
        print("-" * 60)


if __name__ == '__main__':
    main()
