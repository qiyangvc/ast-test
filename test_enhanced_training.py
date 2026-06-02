#!/usr/bin/env python
"""测试增强训练模块：学习率调度器 + 早停法"""
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
from src.enhanced_training import EnhancedTrainer


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


def test_baseline(x_train, y_train, x_val, y_val, x_test, y_test):
    """基线模型：原始训练方式（固定学习率）"""
    logging.info("\n" + "=" * 60)
    logging.info("测试1: 基线模型（原始训练 - 固定学习率）")
    logging.info("=" * 60)

    model = CNNClassifier(
        n_filter=Config.CNN_CONFIG['n_filter'],
        filter_size=Config.CNN_CONFIG['filter_size'],
        stride=Config.CNN_CONFIG['stride'],
        pool_size=Config.CNN_CONFIG['pool_size'],
        pool_strides=Config.CNN_CONFIG['pool_strides'],
        dropout_keep=Config.CNN_CONFIG['dropout_keep']
    )

    optimizer = tf.optimizers.Adam(learning_rate=Config.CNN_CONFIG['learning_rate'])
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'learning_rate': []
    }

    epochs = 10
    batch_size = 16

    for epoch in range(epochs):
        total_loss = 0.0
        total_acc = 0.0
        n_iter = 0
        indices = np.random.permutation(len(x_train))

        for start in range(0, len(x_train), batch_size):
            end = min(start + batch_size, len(x_train))
            batch_indices = indices[start:end]
            batch_x = x_train[batch_indices].astype(np.float32)
            batch_y = y_train[batch_indices].astype(np.int32)

            with tf.GradientTape() as tape:
                logits = model(batch_x, training=True)
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=batch_y, logits=logits))

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            total_loss += loss.numpy()
            correct_pred = tf.equal(tf.argmax(logits, axis=1), batch_y)
            acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32)).numpy()
            total_acc += acc
            n_iter += 1

        train_loss = total_loss / n_iter
        train_acc = total_acc / n_iter

        val_logits = model(x_val.astype(np.float32), training=False)
        val_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_val.astype(np.int32), logits=val_logits)).numpy()
        val_acc = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(val_logits, axis=1), y_val.astype(np.int32)),
            tf.float32)).numpy()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(Config.CNN_CONFIG['learning_rate'])

        logging.info(f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {Config.CNN_CONFIG['learning_rate']:.6f}")

    test_logits = model(x_test.astype(np.float32), training=False)
    test_acc = tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(test_logits, axis=1), y_test.astype(np.int32)),
        tf.float32)).numpy()

    logging.info(f"基线模型测试准确率: {test_acc:.4f}")
    return test_acc, history


def test_enhanced(x_train, y_train, x_val, y_val, x_test, y_test):
    """增强模型：学习率调度器 + 早停法"""
    logging.info("\n" + "=" * 60)
    logging.info("测试2: 增强模型（余弦退火学习率 + 早停法）")
    logging.info("=" * 60)

    model = CNNClassifier(
        n_filter=Config.CNN_CONFIG['n_filter'],
        filter_size=Config.CNN_CONFIG['filter_size'],
        stride=Config.CNN_CONFIG['stride'],
        pool_size=Config.CNN_CONFIG['pool_size'],
        pool_strides=Config.CNN_CONFIG['pool_strides'],
        dropout_keep=Config.CNN_CONFIG['dropout_keep']
    )

    trainer = EnhancedTrainer(
        model=model,
        scheduler_type='cosine',
        warmup_ratio=0.1,
        early_stopping_patience=5
    )
    trainer.compile_model(initial_lr=Config.CNN_CONFIG['learning_rate'])
    history = trainer.fit(
        x_train, y_train, x_val, y_val,
        epochs=15,
        batch_size=16,
        display_step=50
    )

    test_logits = model(x_test.astype(np.float32), training=False)
    test_acc = tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(test_logits, axis=1), y_test.astype(np.int32)),
        tf.float32)).numpy()

    logging.info(f"增强模型测试准确率: {test_acc:.4f}")
    return test_acc, history


def main():
    setup_logging()
    Config.ensure_dirs()

    logging.info("=" * 60)
    logging.info("增强训练模块测试")
    logging.info("创新点1: 学习率调度器（余弦退火 + 预热）")
    logging.info("创新点2: 早停法")
    logging.info("=" * 60)

    x_train, y_train, x_val, y_val, x_test, y_test = load_data()

    baseline_acc, baseline_history = test_baseline(x_train, y_train, x_val, y_val, x_test, y_test)

    enhanced_acc, enhanced_history = test_enhanced(x_train, y_train, x_val, y_val, x_test, y_test)

    logging.info("\n" + "=" * 60)
    logging.info("对比结果")
    logging.info("=" * 60)
    logging.info(f"基线模型准确率:   {baseline_acc:.4f}")
    logging.info(f"增强模型准确率:   {enhanced_acc:.4f}")
    improvement = (enhanced_acc - baseline_acc) * 100
    if improvement > 0:
        logging.info(f"准确率提升:       +{improvement:.2f}%")
    else:
        logging.info(f"准确率变化:       {improvement:.2f}%")

    logging.info("\n学习率变化对比:")
    logging.info("  基线模型: 固定学习率 = " + str(Config.CNN_CONFIG['learning_rate']))
    logging.info("  增强模型学习率变化:")
    for i, lr in enumerate(enhanced_history['learning_rate']):
        logging.info(f"    Epoch {i+1}: LR = {lr:.6f}")


if __name__ == '__main__':
    main()
