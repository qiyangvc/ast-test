"""测试预测功能"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.serving import TextClassifierService
from src.utils import setup_logging

setup_logging()

print("=" * 60)
print("测试预测功能")
print("=" * 60)

test_texts = [
    '免费领取手机话费充值卡',
    '今天天气不错',
    '恭喜您中了大奖，请点击链接领取',
    '明天一起吃饭吧',
    '诈骗短信汇款'
]

for model_type in ['rnn', 'mlp', 'cnn']:
    print(f"\n{'=' * 60}")
    print(f"测试 {model_type.upper()} 模型")
    print('=' * 60)

    service = TextClassifierService(model_type=model_type)

    for text in test_texts:
        result = service.predict(text)
        print(f"\n文本: {text}")
        print(f"预测结果: {result['label']}")
        print(f"置信度: {result['confidence']:.4f}")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)