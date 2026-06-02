"""增强训练模块：学习率调度器 + 早停法"""
import logging
import numpy as np
import tensorflow as tf
from typing import Optional, Callable


class CosineAnnealingScheduler:
    """余弦退火学习率调度器"""

    def __init__(self, initial_lr: float, total_steps: int, warmup_steps: int = 0):
        self.initial_lr = initial_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            return self.initial_lr * (self.current_step + 1) / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.initial_lr * 0.5 * (1 + np.cos(np.pi * progress))

    def step(self):
        self.current_step += 1


class ExponentialDecayScheduler:
    """指数衰减学习率调度器"""

    def __init__(self, initial_lr: float, decay_steps: int, decay_rate: float = 0.96):
        self.initial_lr = initial_lr
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.current_step = 0

    def get_lr(self) -> float:
        return self.initial_lr * (self.decay_rate ** (self.current_step / self.decay_steps))

    def step(self):
        self.current_step += 1


class EarlyStopping:
    """早停法"""

    def __init__(self, patience: int = 5, min_delta: float = 0.0001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode not in ['min', 'max']:
            raise ValueError("mode must be 'min' or 'max'")

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False


class EnhancedTrainer:
    """增强版训练器：支持学习率调度 + 早停"""

    def __init__(self, model: tf.keras.Model, scheduler_type: str = 'cosine',
                 warmup_ratio: float = 0.1, early_stopping_patience: int = 10):
        self.model = model
        self.scheduler_type = scheduler_type
        self.warmup_ratio = warmup_ratio
        self.early_stopping = EarlyStopping(patience=early_stopping_patience, mode='max')
        self.scheduler = None
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rate': []
        }

    def compile_model(self, initial_lr: float):
        self.optimizer = tf.optimizers.Adam(learning_rate=initial_lr)
        self.initial_lr = initial_lr

    def fit(self, x_train, y_train, x_val, y_val, epochs: int, batch_size: int,
            display_step: int = 1, verbose: bool = True):
        """训练模型"""
        n_samples = len(x_train)
        n_val_samples = len(x_val)

        total_steps = (n_samples // batch_size) * epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        if self.scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingScheduler(
                self.initial_lr, total_steps, warmup_steps
            )
        else:
            self.scheduler = ExponentialDecayScheduler(
                self.initial_lr, n_samples // batch_size
            )

        logging.info(f"=" * 60)
        logging.info(f"增强训练配置:")
        logging.info(f"  学习率调度: {self.scheduler_type}")
        logging.info(f"  初始学习率: {self.initial_lr}")
        logging.info(f"  总步数: {total_steps}")
        logging.info(f"  预热步数: {warmup_steps}")
        logging.info(f"  早停耐心: {self.early_stopping.patience}")
        logging.info(f"=" * 60)

        for epoch in range(epochs):
            total_loss = 0.0
            total_acc = 0.0
            n_iter = 0

            indices = np.random.permutation(n_samples)
            current_lr = self.scheduler.get_lr()
            self.optimizer.learning_rate.assign(current_lr)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]
                batch_x = x_train[batch_indices].astype(np.float32)
                batch_y = y_train[batch_indices].astype(np.int32)

                with tf.GradientTape() as tape:
                    logits = self.model(batch_x, training=True)
                    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=batch_y, logits=logits))

                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                total_loss += loss.numpy()
                correct_pred = tf.equal(tf.argmax(logits, axis=1), batch_y)
                acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32)).numpy()
                total_acc += acc
                n_iter += 1

                self.scheduler.step()

                if n_iter % display_step == 0 and verbose:
                    avg_loss = total_loss / n_iter
                    avg_acc = total_acc / n_iter
                    logging.info(f"Epoch {epoch+1}/{epochs}, Iter {n_iter}, "
                               f"Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, LR: {current_lr:.6f}")

            train_loss = total_loss / n_iter
            train_acc = total_acc / n_iter

            val_logits = self.model(x_val.astype(np.float32), training=False)
            val_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y_val.astype(np.int32), logits=val_logits)).numpy()
            val_acc = tf.reduce_mean(tf.cast(
                tf.equal(tf.argmax(val_logits, axis=1), y_val.astype(np.int32)),
                tf.float32)).numpy()

            current_lr = self.scheduler.get_lr()
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)

            logging.info(f"Epoch {epoch+1} completed - "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}")

            if self.early_stopping(val_acc):
                logging.info(f"早停触发！验证准确率连续{self.early_stopping.patience}个epoch无提升")
                break

        logging.info(f"训练完成！最佳验证准确率: {self.early_stopping.best_score:.4f}")
        return self.history


class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss for addressing class imbalance"""

    def __init__(self, alpha=1.0, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        probs = tf.reduce_max(tf.nn.softmax(y_pred), axis=-1)
        focal_weight = self.alpha * tf.pow(1 - probs, self.gamma)
        loss = focal_weight * cross_entropy
        return tf.reduce_mean(loss)


class ModelEnsemble:
    """模型集成"""

    def __init__(self, models: list):
        self.models = models

    def predict(self, x, weights: Optional[list] = None):
        """加权平均集成预测"""
        if weights is None:
            weights = [1.0 / len(self.models)] * len(self.models)

        all_logits = []
        for model in self.models:
            logits = model(x.astype(np.float32), training=False)
            all_logits.append(logits)

        weighted_logits = sum(w * l for w, l in zip(weights, all_logits))
        return weighted_logits

    def predict_proba(self, x, weights: Optional[list] = None):
        """集成预测概率"""
        logits = self.predict(x, weights)
        return tf.nn.softmax(logits)

    def evaluate(self, x, y):
        """评估集成模型"""
        logits = self.predict(x)
        predictions = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_mean(tf.cast(
            tf.equal(predictions, tf.cast(y, tf.int64)), tf.float32)).numpy()

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.cast(y, tf.int32), logits=logits)).numpy()
        return loss, accuracy
