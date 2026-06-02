"""高级训练扩展：数据增强 + 知识蒸馏 + 对抗训练"""
import logging
import numpy as np
import tensorflow as tf
from typing import List, Optional, Tuple


class EmbeddingAugmentation:
    """嵌入向量数据增强"""

    def __init__(self, embedding_matrix: np.ndarray):
        self.embedding_matrix = embedding_matrix
        self.embedding_dim = embedding_matrix.shape[1]

    def noise_injection(self, embeddings: np.ndarray, noise_factor: float = 0.1):
        """添加噪声"""
        noise = np.random.normal(0, noise_factor, embeddings.shape)
        return embeddings + noise

    def scaling(self, embeddings: np.ndarray, scale_range: Tuple[float, float] = (0.9, 1.1)):
        """缩放嵌入向量"""
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return embeddings * scale

    def word_dropout(self, embeddings: np.ndarray, dropout_prob: float = 0.1):
        """词dropout"""
        mask = np.random.binomial(1, 1 - dropout_prob, embeddings.shape)
        return embeddings * mask

    def shuffle_words(self, embeddings: np.ndarray, shuffle_prob: float = 0.1):
        """词顺序打乱"""
        if np.random.random() < shuffle_prob:
            indices = np.arange(len(embeddings))
            np.random.shuffle(indices)
            return embeddings[indices]
        return embeddings

    def augment(self, embeddings: np.ndarray, augment_types: List[str] = ['noise', 'scaling', 'dropout']):
        """组合增强"""
        augmented = embeddings.copy()
        for aug_type in augment_types:
            if aug_type == 'noise':
                augmented = self.noise_injection(augmented)
            elif aug_type == 'scaling':
                augmented = self.scaling(augmented)
            elif aug_type == 'dropout':
                augmented = self.word_dropout(augmented)
        return augmented


class KnowledgeDistillation:
    """知识蒸馏训练"""

    def __init__(self, teacher_model: tf.keras.Model, student_model: tf.keras.Model,
                 temperature: float = 3.0, alpha: float = 0.7):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha

    def distillation_loss(self, student_logits: tf.Tensor, teacher_logits: tf.Tensor,
                          hard_labels: tf.Tensor):
        """蒸馏损失 = α * 软标签损失 + (1-α) * 硬标签损失"""
        soft_loss = tf.keras.losses.KLDivergence()(
            tf.nn.softmax(teacher_logits / self.temperature),
            tf.nn.softmax(student_logits / self.temperature)
        ) * (self.temperature ** 2)

        hard_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=hard_labels, logits=student_logits
        ))

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

    def train_step(self, batch_x: np.ndarray, batch_y: np.ndarray,
                   optimizer: tf.optimizers.Optimizer):
        """单步蒸馏训练"""
        teacher_logits = self.teacher(batch_x.astype(np.float32), training=False)

        with tf.GradientTape() as tape:
            student_logits = self.student(batch_x.astype(np.float32), training=True)
            loss = self.distillation_loss(student_logits, teacher_logits, batch_y)

        grads = tape.gradient(loss, self.student.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.student.trainable_variables))

        return loss.numpy()


class AdversarialTraining:
    """对抗训练（FGM方法）"""

    def __init__(self, model: tf.keras.Model, epsilon: float = 0.1):
        self.model = model
        self.epsilon = epsilon

    def generate_adversarial_examples(self, batch_x: np.ndarray, batch_y: np.ndarray):
        """生成对抗样本"""
        batch_x_tensor = tf.convert_to_tensor(batch_x.astype(np.float32))
        batch_y_tensor = tf.convert_to_tensor(batch_y.astype(np.int32))

        with tf.GradientTape() as tape:
            tape.watch(batch_x_tensor)
            logits = self.model(batch_x_tensor, training=False)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=batch_y_tensor, logits=logits
            ))

        gradients = tape.gradient(loss, batch_x_tensor)
        gradients = tf.sign(gradients)
        adversarial_x = batch_x_tensor + self.epsilon * gradients

        return adversarial_x.numpy()

    def adversarial_train_step(self, batch_x: np.ndarray, batch_y: np.ndarray,
                               optimizer: tf.optimizers.Optimizer):
        """对抗训练单步"""
        batch_x_tensor = tf.convert_to_tensor(batch_x.astype(np.float32))
        batch_y_tensor = tf.convert_to_tensor(batch_y.astype(np.int32))

        with tf.GradientTape() as tape:
            logits = self.model(batch_x_tensor, training=True)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=batch_y_tensor, logits=logits
            ))

        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        adversarial_x = self.generate_adversarial_examples(batch_x, batch_y)
        adversarial_x_tensor = tf.convert_to_tensor(adversarial_x)

        with tf.GradientTape() as tape:
            logits = self.model(adversarial_x_tensor, training=True)
            adv_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=batch_y_tensor, logits=logits
            ))

        grads = tape.gradient(adv_loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss.numpy(), adv_loss.numpy()


class MixupAugmentation:
    """Mixup数据增强"""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def mixup(self, batch_x: np.ndarray, batch_y: np.ndarray):
        """Mixup增强"""
        batch_size = len(batch_x)
        lam = np.random.beta(self.alpha, self.alpha)

        indices = np.random.permutation(batch_size)
        mixed_x = lam * batch_x + (1 - lam) * batch_x[indices]

        y_one_hot = np.eye(2)[batch_y]
        mixed_y = lam * y_one_hot + (1 - lam) * y_one_hot[indices]

        return mixed_x.astype(np.float32), mixed_y.astype(np.float32)


class CutoutAugmentation:
    """Cutout数据增强"""

    def __init__(self, cutout_length: int = 5, cutout_prob: float = 0.5):
        self.cutout_length = cutout_length
        self.cutout_prob = cutout_prob

    def cutout(self, embeddings: np.ndarray):
        """Cutout增强"""
        augmented = embeddings.copy()
        seq_len = len(embeddings)

        if np.random.random() < self.cutout_prob:
            start = np.random.randint(0, seq_len - self.cutout_length)
            end = start + self.cutout_length
            augmented[start:end] = 0

        return augmented


class AdvancedAugmentationTrainer:
    """高级增强训练器"""

    def __init__(self, model: tf.keras.Model, augment_type: str = 'mixup',
                 teacher_model: Optional[tf.keras.Model] = None):
        self.model = model
        self.augment_type = augment_type
        self.teacher_model = teacher_model

        if augment_type == 'mixup':
            self.augmenter = MixupAugmentation(alpha=0.2)
        elif augment_type == 'cutout':
            self.augmenter = CutoutAugmentation(cutout_length=5)
        elif augment_type == 'adversarial':
            self.adversarial_trainer = AdversarialTraining(model, epsilon=0.1)
        elif augment_type == 'distillation' and teacher_model:
            self.distiller = KnowledgeDistillation(teacher_model, model)

        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

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
        logging.info(f"高级增强训练配置:")
        logging.info(f"  模型: {self.model.name}")
        logging.info(f"  增强类型: {self.augment_type}")
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

                if self.augment_type == 'mixup':
                    mixed_x, mixed_y = self.augmenter.mixup(batch_x, batch_y)
                    with tf.GradientTape() as tape:
                        logits = self.model(mixed_x, training=True)
                        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(
                            mixed_y, logits, from_logits=True
                        ))
                    grads = tape.gradient(loss, self.model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                elif self.augment_type == 'adversarial':
                    loss, adv_loss = self.adversarial_trainer.adversarial_train_step(
                        batch_x, batch_y, optimizer
                    )
                    loss = (loss + adv_loss) / 2
                    logits = self.model(batch_x, training=False)

                elif self.augment_type == 'distillation':
                    loss = self.distiller.train_step(batch_x, batch_y, optimizer)
                    logits = self.model(batch_x, training=False)

                else:
                    with tf.GradientTape() as tape:
                        logits = self.model(batch_x, training=True)
                        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=batch_y, logits=logits
                        ))
                    grads = tape.gradient(loss, self.model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                total_loss += loss
                correct_pred = tf.equal(tf.argmax(logits, axis=1), batch_y)
                acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32)).numpy()
                total_acc += acc
                n_iter += 1

                scheduler.step()

                if n_iter % display_step == 0:
                    avg_loss = total_loss / n_iter
                    avg_acc = total_acc / n_iter
                    logging.info(f"Epoch {epoch+1}/{epochs}, Iter {n_iter}, "
                               f"Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")

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
                logging.info(f"早停触发！最佳验证准确率: {early_stopping.best_score:.4f}")
                break

        return self.history