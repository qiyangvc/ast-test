"""分类器模块"""
import logging
import os
import numpy as np
import tensorflow as tf
from typing import Tuple

from src.config import Config


class BaseClassifier(tf.keras.Model):
    """分类器基类"""

    def __init__(self, name: str = None):
        super().__init__(name=name)

    def train(self, x_train, y_train, x_test, y_test, epochs: int, batch_size: int,
              learning_rate: float, display_step: int = 10):
        """训练模型"""
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        logging.info(f"开始训练 {self.__class__.__name__}...")

        n_samples = len(x_train)
        for epoch in range(epochs):
            total_loss = 0.0
            total_acc = 0.0
            n_iter = 0

            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]
                batch_x = x_train[batch_indices].astype(np.float32)
                batch_y = y_train[batch_indices].astype(np.int32)

                with tf.GradientTape() as tape:
                    logits = self(batch_x, training=True)
                    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=batch_y, logits=logits))

                grads = tape.gradient(loss, self.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.trainable_variables))

                total_loss += loss.numpy()
                correct_pred = tf.equal(tf.argmax(logits, axis=1), batch_y)
                acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32)).numpy()
                total_acc += acc
                n_iter += 1

                if n_iter % display_step == 0:
                    avg_loss = total_loss / n_iter
                    avg_acc = total_acc / n_iter
                    logging.info(f"Epoch {epoch+1}, Iter {n_iter}, Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")

            test_logits = self(x_test.astype(np.float32), training=False)
            test_acc = tf.reduce_mean(tf.cast(
                tf.equal(tf.argmax(test_logits, axis=1), y_test.astype(np.int32)),
                tf.float32)).numpy()
            avg_loss = total_loss / n_iter
            avg_acc = total_acc / n_iter

            logging.info(f"Epoch {epoch+1} completed - Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}, Test Acc: {test_acc:.4f}")

        logging.info(f"{self.__class__.__name__}训练完成")

    def save(self, filepath: str):
        """保存模型"""
        self.save_weights(filepath)
        logging.info(f"模型权重已保存到 {filepath}")

    def load(self, filepath: str):
        """加载模型"""
        self.load_weights(filepath).expect_partial()
        logging.info(f"模型权重已从 {filepath} 加载")


class RNNClassifier(BaseClassifier):
    """RNN分类器"""

    def __init__(self, lstm_units: int = 64, recurrent_dropout: float = 0.2):
        super().__init__(name='rnn_classifier')
        self.lstm = tf.keras.layers.LSTM(
            lstm_units,
            dropout=recurrent_dropout,
            return_sequences=False
        )
        self.dropout = tf.keras.layers.Dropout(recurrent_dropout)
        self.dense = tf.keras.layers.Dense(2)

    def call(self, inputs, training=False):
        """前向传播"""
        lstm_out = self.lstm(inputs, training=training)
        drop_out = self.dropout(lstm_out, training=training)
        logits = self.dense(drop_out)
        return logits


class MLPClassifier(BaseClassifier):
    """MLP分类器"""

    def __init__(self, hidden_units: int = 200, dropout_keep: float = 0.5):
        super().__init__(name='mlp_classifier')
        self.dropout1 = tf.keras.layers.Dropout(1 - dropout_keep)
        self.dense1 = tf.keras.layers.Dense(hidden_units, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(1 - dropout_keep)
        self.dense2 = tf.keras.layers.Dense(2)

    def call(self, inputs, training=False):
        """前向传播"""
        drop1 = self.dropout1(inputs, training=training)
        fc1 = self.dense1(drop1)
        drop2 = self.dropout2(fc1, training=training)
        logits = self.dense2(drop2)
        return logits


class CNNClassifier(BaseClassifier):
    """CNN分类器"""

    def __init__(self, n_filter: int = 6, filter_size: int = 3, stride: int = 2,
                 pool_size: int = 3, pool_strides: int = 3, dropout_keep: float = 0.5):
        super().__init__(name='cnn_classifier')
        self.conv1 = tf.keras.layers.Conv1D(
            filters=n_filter,
            kernel_size=filter_size,
            strides=stride,
            padding='same',
            activation='relu'
        )
        self.pool1 = tf.keras.layers.MaxPool1D(
            pool_size=pool_size,
            strides=pool_strides,
            padding='same'
        )
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(1 - dropout_keep)
        self.dense = tf.keras.layers.Dense(2)

    def call(self, inputs, training=False):
        """前向传播"""
        conv_out = self.conv1(inputs, training=training)
        pool_out = self.pool1(conv_out)
        flatten_out = self.flatten(pool_out)
        drop_out = self.dropout(flatten_out, training=training)
        logits = self.dense(drop_out)
        return logits


def train_rnn_classifier():
    """训练RNN分类器"""
    from src.data_loader import DataLoader

    logging.info("加载序列特征数据...")
    data_loader = DataLoader(Config.OUTPUT_DIR)
    x_train, y_train, x_test, y_test = data_loader.load_classifier_data(
        [os.path.join(Config.OUTPUT_DIR, 'features_seq_20.npz')],
        test_size=Config.CLASSIFIER_CONFIG['test_size']
    )

    model = RNNClassifier(
        lstm_units=Config.RNN_CONFIG['lstm_units'],
        recurrent_dropout=Config.RNN_CONFIG['recurrent_dropout']
    )

    model.train(
        x_train, y_train, x_test, y_test,
        epochs=Config.RNN_CONFIG['n_epoch'],
        batch_size=Config.CLASSIFIER_CONFIG['batch_size'],
        learning_rate=Config.RNN_CONFIG['learning_rate'],
        display_step=Config.CLASSIFIER_CONFIG['display_step']
    )

    model_path = os.path.join(Config.WEIGHTS_DIR, 'rnn_classifier')
    model.save(model_path)
    logging.info(f"RNN分类器已保存到 {model_path}")


def train_mlp_classifier():
    """训练MLP分类器"""
    from src.data_loader import DataLoader

    logging.info("加载MLP特征数据...")
    data_loader = DataLoader(Config.OUTPUT_DIR)
    x_train, y_train, x_test, y_test = data_loader.load_classifier_data(
        [os.path.join(Config.OUTPUT_DIR, 'features_mlp.npz')],
        test_size=Config.CLASSIFIER_CONFIG['test_size']
    )

    model = MLPClassifier(
        hidden_units=Config.MLP_CONFIG['hidden_units'],
        dropout_keep=Config.MLP_CONFIG['dropout_keep']
    )

    model.train(
        x_train, y_train, x_test, y_test,
        epochs=Config.MLP_CONFIG['n_epoch'],
        batch_size=Config.CLASSIFIER_CONFIG['batch_size'],
        learning_rate=Config.MLP_CONFIG['learning_rate'],
        display_step=Config.CLASSIFIER_CONFIG['display_step']
    )

    model_path = os.path.join(Config.WEIGHTS_DIR, 'mlp_classifier')
    model.save(model_path)
    logging.info(f"MLP分类器已保存到 {model_path}")


def train_cnn_classifier():
    """训练CNN分类器"""
    from src.data_loader import DataLoader

    logging.info("加载序列特征数据...")
    data_loader = DataLoader(Config.OUTPUT_DIR)
    x_train, y_train, x_test, y_test = data_loader.load_classifier_data(
        [os.path.join(Config.OUTPUT_DIR, 'features_seq_20.npz')],
        test_size=Config.CLASSIFIER_CONFIG['test_size']
    )

    model = CNNClassifier(
        n_filter=Config.CNN_CONFIG['n_filter'],
        filter_size=Config.CNN_CONFIG['filter_size'],
        stride=Config.CNN_CONFIG['stride'],
        pool_size=Config.CNN_CONFIG['pool_size'],
        pool_strides=Config.CNN_CONFIG['pool_strides'],
        dropout_keep=Config.CNN_CONFIG['dropout_keep']
    )

    model.train(
        x_train, y_train, x_test, y_test,
        epochs=Config.CNN_CONFIG['n_epoch'],
        batch_size=Config.CLASSIFIER_CONFIG['batch_size'],
        learning_rate=Config.CNN_CONFIG['learning_rate'],
        display_step=Config.CLASSIFIER_CONFIG['display_step']
    )

    model_path = os.path.join(Config.WEIGHTS_DIR, 'cnn_classifier')
    model.save(model_path)
    logging.info(f"CNN分类器已保存到 {model_path}")


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.utils import setup_logging

    setup_logging()
    Config.ensure_dirs()

    train_rnn_classifier()
    train_mlp_classifier()
    train_cnn_classifier()