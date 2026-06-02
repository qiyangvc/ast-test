#!/usr/bin/env python
"""测试高级模型架构：多尺度CNN + 注意力机制 + Focal Loss + 模型集成"""
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
from src.advanced_models import (
    MultiScaleTextCNN, AttentionCNN, BiLSTMAttention,
    FocalLoss, ModelEnsemble, AdvancedTrainer
)


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


def test_multiscale_cnn(x_train, y_train, x_val, y_val, x_test, y_test):
    """测试多尺度TextCNN"""
    logging.info("\n" + "=" * 60)
    logging.info("创新点5: 多尺度TextCNN (filter_sizes=[2,3,4,5])")
    logging.info("=" * 60)

    model = MultiScaleTextCNN(
        filter_sizes=[2, 3, 4, 5],
        n_filters=32,
        dropout_keep=0.5
    )

    trainer = AdvancedTrainer(
        model=model,
        loss_type='ce',
        scheduler_type='cosine',
        early_stopping_patience=8
    )
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

    logging.info(f"多尺度TextCNN测试准确率: {test_acc:.4f}")
    return test_acc, model


def test_attention_cnn(x_train, y_train, x_val, y_val, x_test, y_test):
    """测试CNN + 注意力机制"""
    logging.info("\n" + "=" * 60)
    logging.info("创新点6: CNN + 自注意力机制")
    logging.info("=" * 60)

    model = AttentionCNN(
        n_filter=64,
        filter_size=3,
        attention_units=32,
        dropout_keep=0.5
    )

    trainer = AdvancedTrainer(
        model=model,
        loss_type='ce',
        scheduler_type='cosine',
        early_stopping_patience=8
    )
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

    logging.info(f"Attention-CNN测试准确率: {test_acc:.4f}")
    return test_acc, model


def test_bilstm_attention(x_train, y_train, x_val, y_val, x_test, y_test):
    """测试BiLSTM + 注意力机制"""
    logging.info("\n" + "=" * 60)
    logging.info("创新点6: BiLSTM + 自注意力机制")
    logging.info("=" * 60)

    model = BiLSTMAttention(
        lstm_units=64,
        attention_units=32,
        dropout_keep=0.5
    )

    trainer = AdvancedTrainer(
        model=model,
        loss_type='ce',
        scheduler_type='cosine',
        early_stopping_patience=8
    )
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

    logging.info(f"BiLSTM-Attention测试准确率: {test_acc:.4f}")
    return test_acc, model


def test_focal_loss(x_train, y_train, x_val, y_val, x_test, y_test):
    """测试Focal Loss"""
    logging.info("\n" + "=" * 60)
    logging.info("创新点3: Focal Loss (alpha=0.25, gamma=2.0)")
    logging.info("=" * 60)

    model = CNNClassifier(
        n_filter=Config.CNN_CONFIG['n_filter'],
        filter_size=Config.CNN_CONFIG['filter_size'],
        stride=Config.CNN_CONFIG['stride'],
        pool_size=Config.CNN_CONFIG['pool_size'],
        pool_strides=Config.CNN_CONFIG['pool_strides'],
        dropout_keep=Config.CNN_CONFIG['dropout_keep']
    )

    trainer = AdvancedTrainer(
        model=model,
        loss_type='focal',
        scheduler_type='cosine',
        early_stopping_patience=8
    )
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

    logging.info(f"Focal Loss测试准确率: {test_acc:.4f}")
    return test_acc, model


def test_model_ensemble(models: list, x_test, y_test):
    """测试模型集成"""
    logging.info("\n" + "=" * 60)
    logging.info("创新点4: 模型集成（加权平均）")
    logging.info("=" * 60)

    ensemble = ModelEnsemble(models, weights=None)

    loss, acc = ensemble.evaluate(x_test, y_test)
    logging.info(f"集成模型测试准确率: {acc:.4f}")

    vote_preds = ensemble.predict_vote(x_test)
    vote_acc = tf.reduce_mean(tf.cast(
        tf.equal(vote_preds, tf.cast(y_test, tf.int64)), tf.float32)).numpy()
    logging.info(f"投票集成测试准确率: {vote_acc:.4f}")

    return acc, vote_acc


def main():
    setup_logging()
    Config.ensure_dirs()

    logging.info("=" * 60)
    logging.info("高级模型架构测试")
    logging.info("创新点3: Focal Loss")
    logging.info("创新点4: 模型集成")
    logging.info("创新点5: 多尺度TextCNN")
    logging.info("创新点6: 注意力机制")
    logging.info("=" * 60)

    x_train, y_train, x_val, y_val, x_test, y_test = load_data()

    results = {}
    models = []

    acc_focal, model_focal = test_focal_loss(x_train, y_train, x_val, y_val, x_test, y_test)
    results['Focal Loss'] = acc_focal
    models.append(model_focal)

    acc_multiscale, model_multiscale = test_multiscale_cnn(x_train, y_train, x_val, y_val, x_test, y_test)
    results['MultiScale-CNN'] = acc_multiscale
    models.append(model_multiscale)

    acc_attention_cnn, model_attention_cnn = test_attention_cnn(x_train, y_train, x_val, y_val, x_test, y_test)
    results['Attention-CNN'] = acc_attention_cnn
    models.append(model_attention_cnn)

    acc_bilstm, model_bilstm = test_bilstm_attention(x_train, y_train, x_val, y_val, x_test, y_test)
    results['BiLSTM-Attention'] = acc_bilstm
    models.append(model_bilstm)

    ensemble_acc, vote_acc = test_model_ensemble(models, x_test, y_test)
    results['Ensemble-Weighted'] = ensemble_acc
    results['Ensemble-Vote'] = vote_acc

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
    best_model = max(results, key=results.get)
    best_acc = results[best_model]
    logging.info(f"最佳模型: {best_model} ({best_acc:.4f})")


if __name__ == '__main__':
    main()