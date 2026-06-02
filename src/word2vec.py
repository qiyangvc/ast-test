"""Word2Vec词向量训练模块"""
import logging
import os
import collections
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.models import Model
from tensorlayer.layers import Embedding, Dense
from typing import List

from src.config import Config
from src.utils import setup_logging


def build_vocab_with_min_freq(words: List[str], min_freq: int = 1) -> dict:
    """构建词汇表，过滤低频词"""
    counter = collections.Counter(words)
    vocab = {}
    vocab['UNK'] = 0
    idx = 1
    for word, count in counter.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab


class SimpleWord2Vec(Model):
    """简化的Word2Vec模型"""
    
    def __init__(self, vocab_size: int, embedding_size: int = 200):
        super().__init__()
        self.embedding = Embedding(vocabulary_size=vocab_size, embedding_size=embedding_size)
        self.dense = Dense(n_units=vocab_size, in_channels=embedding_size)
    
    def forward(self, inputs):
        """前向传播 - 输入是中心词ID"""
        embed = self.embedding(inputs)
        logits = self.dense(embed)
        return logits


class Word2VecTrainer:
    """Word2Vec训练器"""
    
    def __init__(self):
        self.config = Config.WORD2VEC_CONFIG
        self.vocab_size = 0
        self.dictionary = None
        self.reverse_dictionary = None
        self.optimizer = None
        self.embeddings = None
        self.nce_weights = None
        self.nce_biases = None
    
    def prepare_data(self, words: List[str]) -> None:
        """准备训练数据"""
        self.dictionary = build_vocab_with_min_freq(words, self.config['min_freq'])
        self.vocab_size = len(self.dictionary)
        self.reverse_dictionary = {v: k for k, v in self.dictionary.items()}
        
        vocab_path = os.path.join(Config.WEIGHTS_DIR, 'vocab.txt')
        with open(vocab_path, 'w', encoding='utf-8') as f:
            for word, idx in self.dictionary.items():
                f.write(f"{word}\t{idx}\n")
        
        logging.info(f"词汇表大小: {self.vocab_size}")
        
        embedding_size = self.config['embedding_size']
        self.embeddings = tf.Variable(
            tf.random.uniform([self.vocab_size, embedding_size], -1.0, 1.0)
        )
        self.nce_weights = tf.Variable(
            tf.random.truncated_normal([self.vocab_size, embedding_size],
                                      stddev=1.0 / tf.sqrt(tf.cast(embedding_size, tf.float32)))
        )
        self.nce_biases = tf.Variable(tf.zeros([self.vocab_size]))
        
        self.optimizer = tf.optimizers.Adam(learning_rate=self.config['learning_rate'])
    
    def generate_training_data(self, words: List[str], window_size: int = 2) -> tuple:
        """生成skip-gram训练数据"""
        data = []
        for i, word in enumerate(words):
            if word in self.dictionary:
                center_id = self.dictionary[word]
                for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
                    if i != j and words[j] in self.dictionary:
                        context_id = self.dictionary[words[j]]
                        data.append((center_id, context_id))
        return data
    
    def train(self, words: List[str]) -> None:
        """训练Word2Vec模型"""
        self.prepare_data(words)
        
        training_data = self.generate_training_data(words, self.config['skip_window'])
        
        if len(training_data) == 0:
            logging.warning("没有生成训练数据，使用随机词向量")
            self._save_random_embeddings()
            return
        
        logging.info(f"训练数据量: {len(training_data)}")
        logging.info("开始训练Word2Vec模型...")
        
        train_inputs_np = np.asarray([d[0] for d in training_data], dtype=np.int64)
        train_labels_np = np.expand_dims(np.asarray([d[1] for d in training_data], dtype=np.int64), axis=-1)
        
        for epoch in range(self.config['n_epoch']):
            total_loss = 0.0
            n_iter = 0
            
            batch_size = min(self.config['batch_size'], len(training_data))
            num_batches = len(training_data) // batch_size
            
            indices = np.random.permutation(len(training_data))
            
            for start in range(0, len(training_data), batch_size):
                end = min(start + batch_size, len(training_data))
                batch_indices = indices[start:end]
                
                batch_inputs = tf.convert_to_tensor(train_inputs_np[batch_indices], dtype=tf.int64)
                batch_labels = tf.convert_to_tensor(train_labels_np[batch_indices], dtype=tf.int64)
                
                with tf.GradientTape() as tape:
                    embed = tf.nn.embedding_lookup(self.embeddings, batch_inputs)
                    loss = tf.reduce_mean(tf.nn.nce_loss(
                        weights=self.nce_weights,
                        biases=self.nce_biases,
                        labels=batch_labels,
                        inputs=embed,
                        num_sampled=self.config['num_sampled'],
                        num_classes=self.vocab_size
                    ))
                
                grads = tape.gradient(loss, [self.embeddings, self.nce_weights, self.nce_biases])
                self.optimizer.apply_gradients(zip(grads, [self.embeddings, self.nce_weights, self.nce_biases]))
                
                total_loss += loss.numpy()
                n_iter += 1
                
                if n_iter % self.config['print_freq'] == 0:
                    avg_loss = total_loss / n_iter
                    logging.info(f"Epoch {epoch+1}, Iter {n_iter}/{num_batches}, Loss: {avg_loss:.4f}")
            
            if n_iter > 0:
                avg_loss = total_loss / n_iter
                logging.info(f"Epoch {epoch+1} completed, Average Loss: {avg_loss:.4f}")
        
        logging.info("Word2Vec训练完成")
        
        self._save_embeddings()
    
    def _save_embeddings(self):
        """保存词向量"""
        weights_path = os.path.join(Config.WEIGHTS_DIR, self.config['model_name'])
        
        embeddings = {}
        embedding_weights = self.embeddings.numpy()
        
        for word, idx in self.dictionary.items():
            embeddings[word] = embedding_weights[idx]
        
        np.save(weights_path + '.npy', embeddings)
        logging.info(f"词向量已保存到 {weights_path}.npy")
    
    def _save_random_embeddings(self):
        """保存随机词向量（用于测试）"""
        weights_path = os.path.join(Config.WEIGHTS_DIR, self.config['model_name'])
        
        vocab_size = len(self.dictionary)
        embedding_size = self.config['embedding_size']
        
        random_embeddings = np.random.randn(vocab_size, embedding_size).astype(np.float32)
        
        embeddings = {}
        for word, idx in self.dictionary.items():
            embeddings[word] = random_embeddings[idx]
        
        np.save(weights_path + '.npy', embeddings)
        logging.info(f"随机词向量已保存到 {weights_path}.npy")


def train_word2vec():
    """训练Word2Vec词向量"""
    from src.data_loader import DataLoader
    
    setup_logging()
    Config.ensure_dirs()
    
    data_loader = DataLoader(Config.MSG_LOG_DIR)
    words = data_loader.load_word2vec_data()
    
    trainer = Word2VecTrainer()
    trainer.train(words)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    train_word2vec()