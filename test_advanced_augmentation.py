#!/usr/bin/env python
"""测试高级增强技术：Mixup + 对抗训练 + 知识蒸馏"""
import os
import sys
import logging
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.utils import setup_logging
from src.data_loader import DataLoader
from src.classifier import CNNClassifier
from src.advanced_models import BiLSTMAttention
from src.advanced_augmentation import AdvancedAugmentationTrainer


def load_data():
    """加载数据"""
    logging.info("加载序列特征数据...")
    data_loader = DataLoader(Config.OUTPUT_DIR)
    x_train, y_train, x_test, y_test = data_loader.load_classifier_data(
        [os.path.join(Config.OUTPUT_DIR, 'features_seq_20.npz')],
        test_size=Config.CLASSIFIER_CONFIG['test_size']
    )

    val_size = int(len(x_train) * 0.15)
    x_val = x_train[:val_size]
    y_val = y_train[:val_size]
    x_train_sub = x_train[val_size:]
    y_train_sub = y_train[val_size:]

    logging.info(f"训练集: {len(x_train_sub)} 样本")
    logging.info(f"验证集: {len(x_val)} 样本")
    logging.info(f"测试集: {len(x_test)} 样本")
    return x_train_sub, y_train_sub, x_val, y_val, x_test, y_test


def test_mixup(x_train, y_train, x_val, y_val, x_test, y_test):
    """测试Mixup数据增强"""
    logging.info("\n" + "=" * 60)
    logging.info("创新点7: Mixup数据增强 (alpha=0.2)")
    logging.info("=" * 60)

    model = CNNClassifier(
        n_filter=Config.CNN_CONFIG['n_filter'],
        filter_size=Config.CNN_CONFIG['filter_size'],
        stride=Config.CNN_CONFIG['stride'],
        pool_size=Config.CNN_CONFIG['pool_size'],
        pool_strides=Config.CNN_CONFIG['pool_strides'],
        dropout_keep=Config.CNN_CONFIG['dropout_keep']
    )

    trainer = AdvancedAugmentationTrainer(model, augment_type='mixup')
    history = trainer.fit(
        x_train, y_train, x_val, y_val,
        epochs=15,
        batch_size=16,
        learning_rate=0.001,
        display_step=100
    )

    test_logits = model(x_test.astype(np.float32), training=False)
    test_acc = tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(test_logits, axis=1), y_test.astype(np.int32)),
        tf.float32)).numpy()

    logging.info(f"Mixup增强测试准确率: {test_acc:.4f}")
    return test_acc, model


def test_adversarial(x_train, y_train, x_val, y_val, x_test, y_test):
    """测试对抗训练"""
    logging.info("\n" + "=" * 60)
    logging.info("创新点9: 对抗训练 (FGM, epsilon=0.1)")
    logging.info("=" * 60)

    model = CNNClassifier(
        n_filter=Config.CNN_CONFIG['n_filter'],
        filter_size=Config.CNN_CONFIG['filter_size'],
        stride=Config.CNN_CONFIG['stride'],
        pool_size=Config.CNN_CONFIG['pool_size'],
        pool_strides=Config.CNN_CONFIG['pool_strides'],
        dropout_keep=Config.CNN_CONFIG['dropout_keep']
    )

    trainer = AdvancedAugmentationTrainer(model, augment_type='adversarial')
    history = trainer.fit(
        x_train, y_train, x_val, y_val,
        epochs=15,
        batch_size=16,
        learning_rate=0.001,
        display_step=100
    )

    test_logits = model(x_test.astype(np.float32), training=False)
    test_acc = tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(test_logits, axis=1), y_test.astype(np.int32)),
        tf.float32)).numpy()

    logging.info(f"对抗训练测试准确率: {test_acc:.4f}")
    return test_acc, model


def test_distillation(x_train, y_train, x_val, y_val, x_test, y_test, teacher_model):
    """测试知识蒸馏"""
    logging.info("\n" + "=" * 60)
    logging.info("创新点8: 知识蒸馏 (temperature=3.0, alpha=0.7)")
    logging.info("=" * 60)

    student_model = CNNClassifier(
        n_filter=16,
        filter_size=Config.CNN_CONFIG['filter_size'],
        stride=Config.CNN_CONFIG['stride'],
        pool_size=Config.CNN_CONFIG['pool_size'],
        pool_strides=Config.CNN_CONFIG['pool_strides'],
        dropout_keep=Config.CNN_CONFIG['dropout_keep']
    )

    trainer = AdvancedAugmentationTrainer(
        student_model,
        augment_type='distillation',
        teacher_model=teacher_model
    )
    history = trainer.fit(
        x_train, y_train, x_val, y_val,
        epochs=15,
        batch_size=16,
        learning_rate=0.001,
        display_step=100
    )

    test_logits = student_model(x_test.astype(np.float32), training=False)
    test_acc = tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(test_logits, axis=1), y_test.astype(np.int32)),
        tf.float32)).numpy()

    logging.info(f"知识蒸馏测试准确率: {test_acc:.4f}")
    return test_acc


def main():
    setup_logging()
    Config.ensure_dirs()

    logging.info("=" * 60)
    logging.info("高级增强技术测试")
    logging.info("创新点7: Mixup数据增强")
    logging.info("创新点8: 知识蒸馏")
    logging.info("创新点9: 对抗训练")
    logging.info("=" * 60)

    x_train, y_train, x_val, y_val, x_test, y_test = load_data()

    results = {}

    mixup_acc, mixup_model = test_mixup(x_train, y_train, x_val, y_val, x_test, y_test)
    results['Mixup'] = mixup_acc

    adversarial_acc, adv_model = test_adversarial(x_train, y_train, x_val, y_val, x_test, y_test)
    results['Adversarial'] = adversarial_acc

    distillation_acc = test_distillation(x_train, y_train, x_val, y_val, x_test, y_test, adv_model)
    results['Distillation'] = distillation_acc

    logging.info("\n" + "=" * 60)
    logging.info("最终对比结果")
    logging.info("=" * 60)

    baseline_acc = 0.9465
    logging.info(f"基线模型准确率:       {baseline_acc:.4f}")
    logging.info("-" * 40)

    for name, acc in results.items():
        improvement = (acc - baseline_acc) * 100
        logging.info(f"{name}: {acc:.4f} ({improvement:+.2f}%)")

    logging.info("-" * 40)
    best_method = max(results, key=results.get)
    best_acc = results[best_method]
    logging.info(f"最佳增强方法: {best_method} ({best_acc:.4f})")


if __name__ == '__main__':
    main()