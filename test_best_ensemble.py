#!/usr/bin/env python
"""测试最佳模型集成：BiLSTM-Attention + MultiScale-CNN"""
import os
import sys
import logging
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.utils import setup_logging
from src.data_loader import DataLoader
from src.advanced_models import BiLSTMAttention, MultiScaleTextCNN
from src.advanced_optimizers import LabelSmoothingTrainer, BestModelEnsemble


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


def test_best_ensemble(x_train, y_train, x_val, y_val, x_test, y_test):
    """测试最佳模型集成"""
    logging.info("\n" + "=" * 60)
    logging.info("创新点13: 最佳模型集成 (BiLSTM-Attention + MultiScale-CNN)")
    logging.info("=" * 60)

    logging.info("训练BiLSTM-Attention模型...")
    bilstm_model = BiLSTMAttention(
        lstm_units=128,
        attention_units=64,
        dropout_keep=Config.CNN_CONFIG['dropout_keep']
    )
    bilstm_trainer = LabelSmoothingTrainer(bilstm_model, smoothing=0.1)
    bilstm_trainer.fit(
        x_train, y_train, x_val, y_val,
        epochs=15,
        batch_size=16,
        learning_rate=0.001,
        display_step=100
    )
    bilstm_acc = tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(bilstm_model(x_test.astype(np.float32), training=False), axis=1),
                y_test.astype(np.int32)), tf.float32)).numpy()
    logging.info(f"BiLSTM-Attention测试准确率: {bilstm_acc:.4f}")

    logging.info("训练MultiScale-CNN模型...")
    multiscale_model = MultiScaleTextCNN(
        filter_sizes=[2, 3, 4, 5],
        n_filters=32,
        dropout_keep=Config.CNN_CONFIG['dropout_keep']
    )
    multiscale_trainer = LabelSmoothingTrainer(multiscale_model, smoothing=0.1)
    multiscale_trainer.fit(
        x_train, y_train, x_val, y_val,
        epochs=15,
        batch_size=16,
        learning_rate=0.001,
        display_step=100
    )
    multiscale_acc = tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(multiscale_model(x_test.astype(np.float32), training=False), axis=1),
                y_test.astype(np.int32)), tf.float32)).numpy()
    logging.info(f"MultiScale-CNN测试准确率: {multiscale_acc:.4f}")

    logging.info("构建最佳模型集成...")
    ensemble = BestModelEnsemble(
        models=[bilstm_model, multiscale_model],
        model_names=['BiLSTM-Attention', 'MultiScale-CNN']
    )
    ensemble_loss, ensemble_acc = ensemble.evaluate(x_test, y_test)

    logging.info(f"最佳模型集成测试准确率: {ensemble_acc:.4f}")
    return ensemble_acc, ensemble


def main():
    setup_logging()
    Config.ensure_dirs()

    logging.info("=" * 60)
    logging.info("最佳模型集成测试")
    logging.info("创新点13: BiLSTM-Attention + MultiScale-CNN 集成")
    logging.info("=" * 60)

    x_train, y_train, x_val, y_val, x_test, y_test = load_data()

    ensemble_acc, _ = test_best_ensemble(x_train, y_train, x_val, y_val, x_test, y_test)

    logging.info("\n" + "=" * 60)
    logging.info("最终结果")
    logging.info("=" * 60)

    baseline_acc = 0.9465
    best_previous = 0.9673

    logging.info(f"基线模型准确率:       {baseline_acc:.4f}")
    logging.info(f"最佳单项模型准确率:   {best_previous:.4f} (BiLSTM-Attention)")
    logging.info("-" * 50)

    improvement = (ensemble_acc - baseline_acc) * 100
    best_improvement = (ensemble_acc - best_previous) * 100
    logging.info(f"BestEnsemble: {ensemble_acc:.4f} (vs基线: {improvement:+.2f}%, vs最佳: {best_improvement:+.2f}%)")


if __name__ == '__main__':
    main()