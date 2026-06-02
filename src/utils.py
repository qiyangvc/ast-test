import logging
import os
import tarfile
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from typing import Dict, List, Tuple, Any


def setup_logging(log_level: int = logging.INFO) -> None:
    """配置日志"""
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(format=fmt, level=log_level)


def download_and_extract_data(url: str, data_dir: str, filename: str) -> None:
    """下载并解压数据"""
    if not os.path.exists(os.path.join(data_dir, filename.replace('.tar.gz', ''))):
        tl.files.maybe_download_and_extract(
            filename,
            data_dir,
            url
        )
        tarfile.open(os.path.join(data_dir, filename), 'r').extractall(data_dir)


def load_word_dataset(filepaths: List[str]) -> List[str]:
    """加载分词后的文本数据"""
    words = []
    for filepath in filepaths:
        with open(filepath, encoding='utf-8') as f:
            for line in f:
                for word in line.strip().split(' '):
                    if word != '':
                        words.append(word)
    return words


def get_vocabulary_size(words: List[str], min_freq: int = 3) -> int:
    """获取词频不小于min_freq的单词数量"""
    from collections import Counter
    size = 1  # 为UNK预留
    counts = Counter(words).most_common()
    for word, count in counts:
        if count >= min_freq:
            size += 1
    return size


def save_weights(model: tl.models.Model, filepath: str) -> None:
    """保存模型权重"""
    path = os.path.dirname(os.path.abspath(filepath))
    if not os.path.isdir(path):
        os.makedirs(path)
    model.save_weights(filepath=filepath)


def load_weights(model: tl.models.Model, filepath: str) -> None:
    """加载模型权重"""
    if os.path.isfile(filepath):
        model.load_weights(filepath=filepath)


def save_embedding(dictionary: Dict[str, int], network: tl.layers.Word2vecEmbedding, 
                   filepath: str) -> None:
    """保存词向量"""
    words, ids = zip(*dictionary.items())
    params = network.normalized_embeddings
    embeddings = tf.nn.embedding_lookup(params, tf.constant(ids, dtype=tf.int32))
    wv = dict(zip(words, embeddings))
    path = os.path.dirname(os.path.abspath(filepath))
    if not os.path.isdir(path):
        os.makedirs(path)
    tl.files.save_any_to_npy(save_dict=wv, name=filepath + '.npy')


def load_embedding(filepath: str) -> Dict[str, np.ndarray]:
    """加载词向量"""
    return tl.files.load_npy_to_any(name=filepath)


def accuracy(y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
    """计算准确率"""
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


def pad_sequence(sequence: List[np.ndarray], max_len: int, 
                 pad_value: float = 0.0) -> tf.Tensor:
    """对序列进行padding"""
    pad_len = max_len - len(sequence)
    if pad_len > 0:
        pad_shape = sequence[0].shape if sequence else (200,)
        sequence += [tf.convert_to_tensor(np.full(pad_shape, pad_value), dtype=tf.float32) 
                     for _ in range(pad_len)]
    return tf.convert_to_tensor(sequence, dtype=tf.float32)


def format_batch_for_rnn(batch_x: List[List[np.ndarray]], 
                         batch_y: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:
    """格式化RNN输入批次"""
    batch_y = batch_y.astype(np.int32)
    max_seq_len = max([len(d) for d in batch_x])
    for i, d in enumerate(batch_x):
        batch_x[i] += [tf.convert_to_tensor(np.zeros(200), dtype=tf.float32) 
                       for _ in range(max_seq_len - len(d))]
        batch_x[i] = tf.convert_to_tensor(batch_x[i], dtype=tf.float32)
    batch_x = tf.convert_to_tensor(list(batch_x), dtype=tf.float32)
    return batch_x, batch_y


def format_batch_for_cnn(batch_x: List[List[np.ndarray]], batch_y: np.ndarray, 
                         max_seq_len: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """格式化CNN输入批次"""
    batch_y = batch_y.astype(np.int32)
    for i, d in enumerate(batch_x):
        batch_x[i] = d[:max_seq_len]
        batch_x[i] += [tf.convert_to_tensor(np.zeros(200), dtype=tf.float32) 
                       for _ in range(max_seq_len - len(d))]
        batch_x[i] = tf.convert_to_tensor(batch_x[i], dtype=tf.float32)
    batch_x = tf.convert_to_tensor(list(batch_x), dtype=tf.float32)
    return batch_x, batch_y
