"""高级模型架构：多尺度TextCNN + 注意力机制 + Focal Loss"""
import logging
import numpy as np
import tensorflow as tf
from typing import List, Optional


class MultiScaleTextCNN(tf.keras.Model):
    """多尺度TextCNN：捕获不同n-gram特征"""

    def __init__(self, filter_sizes: List[int] = [2, 3, 4, 5],
                 n_filters: int = 32, dropout_keep: float = 0.5):
        super().__init__(name='multiscale_textcnn')

        self.conv_layers = []
        for filter_size in filter_sizes:
            conv = tf.keras.layers.Conv1D(
                filters=n_filters,
                kernel_size=filter_size,
                strides=1,
                padding='valid',
                activation='relu',
                name=f'conv_{filter_size}'
            )
            self.conv_layers.append(conv)

        self.pool_layers = []
        for i, filter_size in enumerate(filter_sizes):
            pool = tf.keras.layers.GlobalMaxPool1D(name=f'pool_{filter_size}')
            self.pool_layers.append(pool)

        self.concatenate = tf.keras.layers.Concatenate(name='concat')
        self.dropout = tf.keras.layers.Dropout(1 - dropout_keep, name='dropout')
        self.dense = tf.keras.layers.Dense(2, name='output')

    def call(self, inputs, training=False):
        conv_outputs = []
        pool_outputs = []

        for conv, pool in zip(self.conv_layers, self.pool_layers):
            conv_out = conv(inputs, training=training)
            pool_out = pool(conv_out)
            conv_outputs.append(conv_out)
            pool_outputs.append(pool_out)

        concat_out = self.concatenate(pool_outputs)
        drop_out = self.dropout(concat_out, training=training)
        logits = self.dense(drop_out)
        return logits


class SelfAttention(tf.keras.layers.Layer):
    """自注意力机制层"""

    def __init__(self, attention_units: int = 64, return_sequences: bool = False):
        super().__init__()
        self.attention_units = attention_units
        self.return_sequences = return_sequences

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.attention_units),
            initializer='glorot_uniform',
            trainable=True,
            name='attention_W'
        )
        self.b = self.add_weight(
            shape=(self.attention_units,),
            initializer='zeros',
            trainable=True,
            name='attention_b'
        )
        self.u = self.add_weight(
            shape=(self.attention_units, 1),
            initializer='glorot_uniform',
            trainable=True,
            name='attention_u'
        )
        super().build(input_shape)

    def call(self, inputs):
        logits = tf.tensordot(inputs, self.W, axes=[-1, 0]) + self.b
        logits = tf.tanh(logits)
        logits = tf.tensordot(logits, self.u, axes=[-1, 0])
        attention_weights = tf.nn.softmax(logits, axis=1)

        if self.return_sequences:
            weighted = inputs * attention_weights
            return weighted, attention_weights
        else:
            weighted_sum = tf.reduce_sum(inputs * attention_weights, axis=1)
            return weighted_sum, attention_weights


class AttentionCNN(tf.keras.Model):
    """CNN + 注意力机制"""

    def __init__(self, n_filter: int = 64, filter_size: int = 3,
                 attention_units: int = 32, dropout_keep: float = 0.5):
        super().__init__(name='attention_cnn')

        self.conv1 = tf.keras.layers.Conv1D(
            filters=n_filter,
            kernel_size=filter_size,
            strides=1,
            padding='same',
            activation='relu'
        )
        self.attention = SelfAttention(attention_units, return_sequences=False)
        self.dropout = tf.keras.layers.Dropout(1 - dropout_keep)
        self.dense = tf.keras.layers.Dense(2)

    def call(self, inputs, training=False):
        conv_out = self.conv1(inputs, training=training)
        attention_out, attention_weights = self.attention(conv_out)
        drop_out = self.dropout(attention_out, training=training)
        logits = self.dense(drop_out)
        return logits


class BiLSTMAttention(tf.keras.Model):
    """双向LSTM + 注意力机制"""

    def __init__(self, lstm_units: int = 64, attention_units: int = 32,
                 dropout_keep: float = 0.5):
        super().__init__(name='bilstm_attention')

        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_units, return_sequences=True)
        )
        self.attention = SelfAttention(attention_units, return_sequences=False)
        self.dropout = tf.keras.layers.Dropout(1 - dropout_keep)
        self.dense = tf.keras.layers.Dense(2)

    def call(self, inputs, training=False):
        lstm_out = self.bilstm(inputs, training=training)
        attention_out, attention_weights = self.attention(lstm_out)
        drop_out = self.dropout(attention_out, training=training)
        logits = self.dense(drop_out)
        return logits


class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss：处理类别不平衡"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )

        probs = tf.nn.softmax(y_pred)
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        pt = tf.reduce_sum(probs * y_true_one_hot, axis=-1)

        focal_weight = self.alpha * tf.pow(1 - pt, self.gamma)
        focal_loss = focal_weight * ce_loss
        return tf.reduce_mean(focal_loss)


class LabelSmoothingLoss(tf.keras.losses.Loss):
    """标签平滑损失：防止过拟合"""

    def __init__(self, smoothing: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.smoothing = smoothing

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        num_classes = tf.shape(y_pred)[-1]

        smooth_labels = tf.one_hot(y_true, depth=num_classes)
        smooth_labels = smooth_labels * (1 - self.smoothing) + self.smoothing / num_classes

        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=smooth_labels, logits=y_pred
        )
        return tf.reduce_mean(loss)


class ModelEnsemble:
    """模型集成：投票 + 加权平均"""

    def __init__(self, models: List[tf.keras.Model], weights: Optional[List[float]] = None):
        self.models = models
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights

    def predict(self, x):
        """加权平均集成"""
        all_logits = []
        for model in self.models:
            logits = model(x.astype(np.float32), training=False)
            all_logits.append(logits)

        weighted_logits = sum(w * l for w, l in zip(self.weights, all_logits))
        return weighted_logits

    def predict_proba(self, x):
        """集成预测概率"""
        logits = self.predict(x)
        return tf.nn.softmax(logits)

    def predict_vote(self, x):
        """投票集成"""
        all_preds = []
        for model in self.models:
            logits = model(x.astype(np.float32), training=False)
            preds = tf.argmax(logits, axis=1)
            all_preds.append(preds)

        stacked = tf.stack(all_preds, axis=0)
        voted = tf.reduce_mean(tf.cast(stacked, tf.float32), axis=0)
        return tf.cast(tf.round(voted), tf.int64)

    def evaluate(self, x, y):
        """评估集成模型"""
        logits = self.predict(x)
        predictions = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_mean(tf.cast(
            tf.equal(predictions, tf.cast(y, tf.int64)), tf.float32)).numpy()

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.cast(y, tf.int32), logits=logits)).numpy()
        return loss, accuracy


class TextAugmentation:
    """文本数据增强"""

    def __init__(self, embedding_matrix: np.ndarray, word2idx: dict):
        self.embedding_matrix = embedding_matrix
        self.word2idx = word2idx
        self.idx2word = {v: k for k, v in word2idx.items()}

    def random_replace(self, text_indices: List[int], replace_prob: float = 0.1):
        """随机替换同义词"""
        augmented = text_indices.copy()
        for i in range(len(augmented)):
            if np.random.random() < replace_prob:
                if augmented[i] in self.idx2word:
                    similar_words = self._find_similar_words(augmented[i], top_k=5)
                    if similar_words:
                        augmented[i] = np.random.choice(similar_words)
        return augmented

    def random_delete(self, text_indices: List[int], delete_prob: float = 0.1):
        """随机删除词"""
        augmented = [idx for idx in text_indices if np.random.random() > delete_prob]
        if len(augmented) == 0:
            return text_indices
        return augmented

    def _find_similar_words(self, word_idx: int, top_k: int = 5):
        """找相似词"""
        if word_idx >= len(self.embedding_matrix):
            return []

        word_vec = self.embedding_matrix[word_idx]
        similarities = np.dot(self.embedding_matrix, word_vec)
        similarities[word_idx] = -np.inf
        top_indices = np.argsort(similarities)[-top_k:]
        return top_indices.tolist()


class AdvancedTrainer:
    """高级训练器：支持多种损失函数"""

    def __init__(self, model: tf.keras.Model, loss_type: str = 'focal',
                 scheduler_type: str = 'cosine', early_stopping_patience: int = 10):
        self.model = model
        self.loss_type = loss_type
        self.scheduler_type = scheduler_type
        self.early_stopping_patience = early_stopping_patience

        if loss_type == 'focal':
            self.loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        elif loss_type == 'label_smoothing':
            self.loss_fn = LabelSmoothingLoss(smoothing=0.1)
        else:
            self.loss_fn = None

        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rate': []
        }

    def fit(self, x_train, y_train, x_val, y_val, epochs: int, batch_size: int,
            learning_rate: float = 0.001, display_step: int = 50):
        """训练模型"""
        from src.enhanced_training import CosineAnnealingScheduler, EarlyStopping

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        n_samples = len(x_train)
        total_steps = (n_samples // batch_size) * epochs

        scheduler = CosineAnnealingScheduler(learning_rate, total_steps, warmup_steps=int(total_steps * 0.1))
        early_stopping = EarlyStopping(patience=self.early_stopping_patience, mode='max')

        logging.info(f"=" * 60)
        logging.info(f"高级训练配置:")
        logging.info(f"  模型: {self.model.name}")
        logging.info(f"  损失函数: {self.loss_type}")
        logging.info(f"  学习率调度: {self.scheduler_type}")
        logging.info(f"  早停耐心: {self.early_stopping_patience}")
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

                    if self.loss_fn is not None:
                        loss = self.loss_fn(batch_y, logits)
                    else:
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
                    avg_loss = total_loss / n_iter
                    avg_acc = total_acc / n_iter
                    logging.info(f"Epoch {epoch+1}/{epochs}, Iter {n_iter}, "
                               f"Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, LR: {current_lr:.6f}")

            train_loss = total_loss / n_iter
            train_acc = total_acc / n_iter

            val_logits = self.model(x_val.astype(np.float32), training=False)
            if self.loss_fn is not None:
                val_loss = self.loss_fn(y_val.astype(np.int32), val_logits).numpy()
            else:
                val_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=y_val.astype(np.int32), logits=val_logits)).numpy()
            val_acc = tf.reduce_mean(tf.cast(
                tf.equal(tf.argmax(val_logits, axis=1), y_val.astype(np.int32)),
                tf.float32)).numpy()

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)

            logging.info(f"Epoch {epoch+1} completed - "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if early_stopping(val_acc):
                logging.info(f"早停触发！最佳验证准确率: {early_stopping.best_score:.4f}")
                break

        return self.history