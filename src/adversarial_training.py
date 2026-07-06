"""Embedding-level adversarial training for classifier inputs.

The project represents text as NBOW vectors or word-vector sequences before it
enters the classifiers. FGM/PGD can therefore be applied directly to model
inputs without changing the Word2Vec vocabulary. This complements text-level
AST data generation: text AST tests realistic spam evasion, while embedding AST
improves local robustness around each vectorized sample.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import tensorflow as tf


AttackMethod = Literal["fgm", "pgd"]


@dataclass
class AdversarialTrainingConfig:
    """Configuration for input-level adversarial training."""

    method: AttackMethod = "fgm"
    epsilon: float = 0.5
    alpha: float = 0.5
    pgd_steps: int = 3
    pgd_step_size: float = 0.2
    random_start: bool = True
    grad_norm: Literal["l2", "linf"] = "l2"


@dataclass
class TrainingHistory:
    """History values collected during adversarial training."""

    train_loss: List[float] = field(default_factory=list)
    train_clean_loss: List[float] = field(default_factory=list)
    train_adv_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict[str, List[float]]:
        return {
            "train_loss": self.train_loss,
            "train_clean_loss": self.train_clean_loss,
            "train_adv_loss": self.train_adv_loss,
            "train_acc": self.train_acc,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
        }


def sparse_ce_loss(y_true: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
    y_true = tf.cast(y_true, tf.int32)
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=logits))


def batch_accuracy(y_true: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
    preds = tf.argmax(logits, axis=1, output_type=tf.int32)
    return tf.reduce_mean(tf.cast(tf.equal(preds, tf.cast(y_true, tf.int32)), tf.float32))


def normalize_gradient(grad: tf.Tensor, norm_type: str = "l2", eps: float = 1e-12) -> tf.Tensor:
    """Normalize gradients independently for each sample in a batch."""
    if norm_type == "linf":
        return tf.sign(grad)
    axes = tuple(range(1, len(grad.shape)))
    norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=axes, keepdims=True))
    return grad / (norm + eps)


def project_l2(delta: tf.Tensor, epsilon: float, eps: float = 1e-12) -> tf.Tensor:
    axes = tuple(range(1, len(delta.shape)))
    norm = tf.sqrt(tf.reduce_sum(tf.square(delta), axis=axes, keepdims=True))
    factor = tf.minimum(1.0, epsilon / (norm + eps))
    return delta * factor


class InputAdversary:
    """Base class for adversaries that perturb vectorized model inputs."""

    def __init__(self, config: AdversarialTrainingConfig):
        self.config = config

    def perturb(self, model: tf.keras.Model, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError


class FGMInputAdversary(InputAdversary):
    """Fast gradient method on classifier input vectors."""

    def perturb(self, model: tf.keras.Model, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            tape.watch(x)
            logits = model(x, training=True)
            loss = sparse_ce_loss(y, logits)
        grad = tape.gradient(loss, x)
        if grad is None:
            return tf.identity(x)
        direction = normalize_gradient(grad, self.config.grad_norm)
        return tf.stop_gradient(x + self.config.epsilon * direction)


class PGDInputAdversary(InputAdversary):
    """Projected gradient descent adversary on classifier input vectors."""

    def perturb(self, model: tf.keras.Model, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        if self.config.random_start:
            noise = tf.random.normal(tf.shape(x), dtype=x.dtype)
            noise = normalize_gradient(noise, self.config.grad_norm) * self.config.pgd_step_size
            adv_x = x + noise
        else:
            adv_x = tf.identity(x)

        for _ in range(self.config.pgd_steps):
            with tf.GradientTape() as tape:
                tape.watch(adv_x)
                logits = model(adv_x, training=True)
                loss = sparse_ce_loss(y, logits)
            grad = tape.gradient(loss, adv_x)
            if grad is None:
                break
            step = self.config.pgd_step_size * normalize_gradient(grad, self.config.grad_norm)
            adv_x = adv_x + step
            delta = adv_x - x
            if self.config.grad_norm == "linf":
                delta = tf.clip_by_value(delta, -self.config.epsilon, self.config.epsilon)
            else:
                delta = project_l2(delta, self.config.epsilon)
            adv_x = tf.stop_gradient(x + delta)
        return adv_x


def make_adversary(config: AdversarialTrainingConfig) -> InputAdversary:
    if config.method == "fgm":
        return FGMInputAdversary(config)
    if config.method == "pgd":
        return PGDInputAdversary(config)
    raise ValueError(f"Unsupported adversarial method: {config.method}")


class EmbeddingAdversarialTrainer:
    """Train a classifier with clean + adversarial input loss."""

    def __init__(self, model: tf.keras.Model, config: Optional[AdversarialTrainingConfig] = None):
        self.model = model
        self.config = config or AdversarialTrainingConfig()
        self.adversary = make_adversary(self.config)
        self.history = TrainingHistory()

    def train_step(
        self,
        batch_x: np.ndarray,
        batch_y: np.ndarray,
        optimizer: tf.optimizers.Optimizer,
    ) -> Tuple[float, float, float, float]:
        x = tf.convert_to_tensor(batch_x.astype(np.float32), dtype=tf.float32)
        y = tf.convert_to_tensor(batch_y.astype(np.int32), dtype=tf.int32)
        adv_x = self.adversary.perturb(self.model, x, y)

        with tf.GradientTape() as tape:
            clean_logits = self.model(x, training=True)
            adv_logits = self.model(adv_x, training=True)
            clean_loss = sparse_ce_loss(y, clean_logits)
            adv_loss = sparse_ce_loss(y, adv_logits)
            total_loss = (1.0 - self.config.alpha) * clean_loss + self.config.alpha * adv_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        grads_and_vars = [(g, v) for g, v in zip(grads, self.model.trainable_variables) if g is not None]
        optimizer.apply_gradients(grads_and_vars)
        acc = batch_accuracy(y, clean_logits)
        return (
            float(total_loss.numpy()),
            float(clean_loss.numpy()),
            float(adv_loss.numpy()),
            float(acc.numpy()),
        )

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int,
        batch_size: int,
        learning_rate: float = 0.001,
        display_step: int = 50,
    ) -> Dict[str, List[float]]:
        """Run adversarial training. This is not invoked by import-time code."""
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        n_samples = len(x_train)

        logging.info("=" * 60)
        logging.info("Embedding-level adversarial training")
        logging.info("method=%s epsilon=%.4f alpha=%.4f", self.config.method, self.config.epsilon, self.config.alpha)
        logging.info("=" * 60)

        for epoch in range(epochs):
            total_loss = total_clean = total_adv = total_acc = 0.0
            n_iter = 0
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]
                metrics = self.train_step(x_train[batch_indices], y_train[batch_indices], optimizer)
                step_loss, step_clean, step_adv, step_acc = metrics
                total_loss += step_loss
                total_clean += step_clean
                total_adv += step_adv
                total_acc += step_acc
                n_iter += 1

                if display_step and n_iter % display_step == 0:
                    logging.info(
                        "Epoch %d/%d Iter %d - loss %.4f clean %.4f adv %.4f acc %.4f",
                        epoch + 1,
                        epochs,
                        n_iter,
                        total_loss / n_iter,
                        total_clean / n_iter,
                        total_adv / n_iter,
                        total_acc / n_iter,
                    )

            val_loss, val_acc = evaluate_arrays(self.model, x_val, y_val)
            self.history.train_loss.append(total_loss / max(n_iter, 1))
            self.history.train_clean_loss.append(total_clean / max(n_iter, 1))
            self.history.train_adv_loss.append(total_adv / max(n_iter, 1))
            self.history.train_acc.append(total_acc / max(n_iter, 1))
            self.history.val_loss.append(val_loss)
            self.history.val_acc.append(val_acc)

            logging.info(
                "Epoch %d completed - train loss %.4f val loss %.4f val acc %.4f",
                epoch + 1,
                self.history.train_loss[-1],
                val_loss,
                val_acc,
            )

        return self.history.as_dict()


def evaluate_arrays(model: tf.keras.Model, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x_tensor = tf.convert_to_tensor(x.astype(np.float32), dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y.astype(np.int32), dtype=tf.int32)
    logits = model(x_tensor, training=False)
    loss = sparse_ce_loss(y_tensor, logits)
    acc = batch_accuracy(y_tensor, logits)
    return float(loss.numpy()), float(acc.numpy())
