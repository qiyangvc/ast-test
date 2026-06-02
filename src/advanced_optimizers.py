"""高级优化器：SWA + Lookahead + 最佳集成"""
import logging
import numpy as np
import tensorflow as tf
from typing import List, Optional


class SWA:
    """随机权重平均 (Stochastic Weight Averaging)"""

    def __init__(self, model: tf.keras.Model, swa_start: int = 5, swa_freq: int = 1):
        self.model = model
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_weights = None
        self.swa_count = 0

    def update_weights(self):
        """更新SWA权重"""
        if self.swa_count == 0:
            self.swa_weights = [tf.Variable(w.numpy()) for w in self.model.weights]
        else:
            for i, w in enumerate(self.model.weights):
                self.swa_weights[i].assign_add(w.numpy())

        self.swa_count += 1
        logging.info(f"SWA权重更新: 第{self.swa_count}次平均")

    def apply_swa(self):
        """应用SWA权重"""
        if self.swa_count > 0 and self.swa_weights is not None:
            avg_weights = [w / self.swa_count for w in self.swa_weights]
            for i, w in enumerate(self.model.weights):
                w.assign(avg_weights[i])
            logging.info(f"SWA应用完成: {self.swa_count}次平均")

    def should_average(self, epoch: int) -> bool:
        """判断是否应该进行平均"""
        return epoch >= self.swa_start and (epoch - self.swa_start) % self.swa_freq == 0


class Lookahead:
    """Lookahead优化器包装"""

    def __init__(self, optimizer: tf.optimizers.Optimizer, k: int = 5, alpha: float = 0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.step_counter = 0
        self.slow_weights = None

    def _initialize_slow_weights(self):
        """初始化慢权重"""
        if self.slow_weights is None:
            self.slow_weights = [tf.Variable(w.numpy()) for w in self.optimizer.variables()]

    def _update_slow_weights(self):
        """更新慢权重"""
        self._initialize_slow_weights()
        for slow_w, fast_w in zip(self.slow_weights, self.optimizer.variables()):
            slow_w.assign(self.alpha * fast_w.numpy() + (1 - self.alpha) * slow_w.numpy())

    def apply_gradients(self, grads_and_vars):
        """应用梯度"""
        self.optimizer.apply_gradients(grads_and_vars)
        self.step_counter += 1

        if self.step_counter % self.k == 0:
            self._update_slow_weights()

    def reset(self):
        """重置"""
        self.step_counter = 0
        self.slow_weights = None


class SWALossTrainer:
    """SWA + 余弦退火训练器"""

    def __init__(self, model: tf.keras.Model, swa_start: int = 5):
        self.model = model
        self.swa = SWA(model, swa_start=swa_start)
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def fit(self, x_train, y_train, x_val, y_val, epochs: int, batch_size: int,
            learning_rate: float = 0.001, display_step: int = 50):
        """训练模型"""
        from src.enhanced_training import CosineAnnealingScheduler, EarlyStopping

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        n_samples = len(x_train)
        total_steps = (n_samples // batch_size) * epochs

        scheduler = CosineAnnealingScheduler(learning_rate, total_steps, warmup_steps=int(total_steps * 0.1))
        early_stopping = EarlyStopping(patience=10, mode='max')

        logging.info(f"=" * 60)
        logging.info(f"SWA训练配置:")
        logging.info(f"  SWA开始epoch: {self.swa.swa_start}")
        logging.info(f"  学习率调度: 余弦退火")
        logging.info(f"=" * 60)

        for epoch in range(epochs):
            total_loss = 0.0
            total_acc = 0.0
            n_iter = 0

            indices = np.random.permutation(n_samples)
            current_lr = scheduler.get_lr()
            optimizer.learning_rate.assign(current_lr)

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
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                total_loss += loss.numpy()
                correct_pred = tf.equal(tf.argmax(logits, axis=1), batch_y)
                acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32)).numpy()
                total_acc += acc
                n_iter += 1

                scheduler.step()

                if n_iter % display_step == 0:
                    logging.info(f"Epoch {epoch+1}/{epochs}, Iter {n_iter}, "
                              f"Loss: {total_loss/n_iter:.4f}, Acc: {total_acc/n_iter:.4f}, LR: {current_lr:.6f}")

            if self.swa.should_average(epoch):
                self.swa.update_weights()

            train_loss = total_loss / n_iter
            train_acc = total_acc / n_iter

            val_logits = self.model(x_val.astype(np.float32), training=False)
            val_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y_val.astype(np.int32), logits=val_logits)).numpy()
            val_acc = tf.reduce_mean(tf.cast(
                tf.equal(tf.argmax(val_logits, axis=1), y_val.astype(np.int32)),
                tf.float32)).numpy()

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            logging.info(f"Epoch {epoch+1} completed - "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if early_stopping(val_acc):
                logging.info(f"早停触发！")
                break

        self.swa.apply_swa()
        return self.history


class LookaheadTrainer:
    """Lookahead优化器训练器"""

    def __init__(self, model: tf.keras.Model, k: int = 5, alpha: float = 0.5):
        self.model = model
        self.k = k
        self.alpha = alpha
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def fit(self, x_train, y_train, x_val, y_val, epochs: int, batch_size: int,
            learning_rate: float = 0.001, display_step: int = 50):
        """训练模型"""
        from src.enhanced_training import CosineAnnealingScheduler, EarlyStopping

        base_optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        optimizer = Lookahead(base_optimizer, k=self.k, alpha=self.alpha)
        n_samples = len(x_train)
        total_steps = (n_samples // batch_size) * epochs

        scheduler = CosineAnnealingScheduler(learning_rate, total_steps, warmup_steps=int(total_steps * 0.1))
        early_stopping = EarlyStopping(patience=10, mode='max')

        logging.info(f"=" * 60)
        logging.info(f"Lookahead训练配置:")
        logging.info(f"  k (更新周期): {self.k}")
        logging.info(f"  alpha (插值因子): {self.alpha}")
        logging.info(f"=" * 60)

        for epoch in range(epochs):
            total_loss = 0.0
            total_acc = 0.0
            n_iter = 0

            indices = np.random.permutation(n_samples)
            current_lr = scheduler.get_lr()
            base_optimizer.learning_rate.assign(current_lr)

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
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                total_loss += loss.numpy()
                correct_pred = tf.equal(tf.argmax(logits, axis=1), batch_y)
                acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32)).numpy()
                total_acc += acc
                n_iter += 1

                scheduler.step()

                if n_iter % display_step == 0:
                    logging.info(f"Epoch {epoch+1}/{epochs}, Iter {n_iter}, "
                              f"Loss: {total_loss/n_iter:.4f}, Acc: {total_acc/n_iter:.4f}, LR: {current_lr:.6f}")

            train_loss = total_loss / n_iter
            train_acc = total_acc / n_iter

            val_logits = self.model(x_val.astype(np.float32), training=False)
            val_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y_val.astype(np.int32), logits=val_logits)).numpy()
            val_acc = tf.reduce_mean(tf.cast(
                tf.equal(tf.argmax(val_logits, axis=1), y_val.astype(np.int32)),
                tf.float32)).numpy()

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            logging.info(f"Epoch {epoch+1} completed - "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if early_stopping(val_acc):
                logging.info(f"早停触发！")
                break

        return self.history


class BestModelEnsemble:
    """最佳模型集成：BiLSTM-Attention + MultiScale-CNN"""

    def __init__(self, models: List[tf.keras.Model], model_names: List[str]):
        self.models = models
        self.model_names = model_names

    def predict(self, x):
        """加权预测（根据各模型性能加权）"""
        all_logits = []
        for model in self.models:
            logits = model(x.astype(np.float32), training=False)
            all_logits.append(logits)

        weighted_logits = sum(all_logits) / len(all_logits)
        return weighted_logits

    def predict_proba(self, x):
        """预测概率"""
        logits = self.predict(x)
        return tf.nn.softmax(logits)

    def evaluate(self, x, y):
        """评估"""
        logits = self.predict(x)
        predictions = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_mean(tf.cast(
            tf.equal(predictions, tf.cast(y, tf.int64)), tf.float32)).numpy()

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.cast(y, tf.int32), logits=logits)).numpy()
        return loss, accuracy


class LabelSmoothingLoss(tf.keras.losses.Loss):
    """标签平滑损失"""

    def __init__(self, smoothing: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.smoothing = smoothing

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        num_classes = tf.shape(y_pred)[-1]

        smooth_labels = tf.one_hot(y_true, depth=num_classes)
        smoothing = tf.cast(self.smoothing, tf.float32)
        num_classes_float = tf.cast(num_classes, tf.float32)
        smooth_labels = smooth_labels * (1.0 - smoothing) + smoothing / num_classes_float

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=smooth_labels, logits=y_pred)
        return tf.reduce_mean(loss)


class LabelSmoothingTrainer:
    """标签平滑训练器"""

    def __init__(self, model: tf.keras.Model, smoothing: float = 0.1):
        self.model = model
        self.smoothing = smoothing
        self.loss_fn = LabelSmoothingLoss(smoothing=smoothing)
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def fit(self, x_train, y_train, x_val, y_val, epochs: int, batch_size: int,
            learning_rate: float = 0.001, display_step: int = 50):
        """训练模型"""
        from src.enhanced_training import CosineAnnealingScheduler, EarlyStopping

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        n_samples = len(x_train)
        total_steps = (n_samples // batch_size) * epochs

        scheduler = CosineAnnealingScheduler(learning_rate, total_steps, warmup_steps=int(total_steps * 0.1))
        early_stopping = EarlyStopping(patience=10, mode='max')

        logging.info(f"=" * 60)
        logging.info(f"Label Smoothing训练配置:")
        logging.info(f"  smoothing: {self.smoothing}")
        logging.info(f"=" * 60)

        for epoch in range(epochs):
            total_loss = 0.0
            total_acc = 0.0
            n_iter = 0

            indices = np.random.permutation(n_samples)
            current_lr = scheduler.get_lr()
            optimizer.learning_rate.assign(current_lr)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]
                batch_x = x_train[batch_indices].astype(np.float32)
                batch_y = y_train[batch_indices].astype(np.int32)

                with tf.GradientTape() as tape:
                    logits = self.model(batch_x, training=True)
                    loss = self.loss_fn(batch_y, logits)

                grads = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                total_loss += loss.numpy()
                correct_pred = tf.equal(tf.argmax(logits, axis=1), batch_y)
                acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32)).numpy()
                total_acc += acc
                n_iter += 1

                scheduler.step()

                if n_iter % display_step == 0:
                    logging.info(f"Epoch {epoch+1}/{epochs}, Iter {n_iter}, "
                              f"Loss: {total_loss/n_iter:.4f}, Acc: {total_acc/n_iter:.4f}, LR: {current_lr:.6f}")

            train_loss = total_loss / n_iter
            train_acc = total_acc / n_iter

            val_logits = self.model(x_val.astype(np.float32), training=False)
            val_loss = self.loss_fn(y_val.astype(np.int32), val_logits).numpy()
            val_acc = tf.reduce_mean(tf.cast(
                tf.equal(tf.argmax(val_logits, axis=1), y_val.astype(np.int32)),
                tf.float32)).numpy()

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            logging.info(f"Epoch {epoch+1} completed - "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if early_stopping(val_acc):
                logging.info(f"早停触发！")
                break

        return self.history