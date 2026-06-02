import os
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any

class DataLoader:
    """数据加载器"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
    
    def load_word2vec_data(self) -> Tuple[list, dict]:
        """加载词向量训练数据"""
        from src.utils import load_word_dataset
        
        files = [
            os.path.join(self.data_dir, 'msgpass.log.seg'),
            os.path.join(self.data_dir, 'msgspam.log.seg')
        ]
        
        words = load_word_dataset(files)
        return words
    
    def load_classifier_data(self, files: list, test_size: float = 0.2, 
                            allow_pickle: bool = True) -> Tuple[np.ndarray, np.ndarray, 
                                                               np.ndarray, np.ndarray]:
        """加载分类器训练数据"""
        x = []
        y = []
        
        for file in files:
            data = np.load(file, allow_pickle=allow_pickle)
            if not x or not y:
                x = data['x']
                y = data['y']
            else:
                x = np.append(x, data['x'], axis=0)
                y = np.append(y, data['y'], axis=0)
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
        return x_train, y_train, x_test, y_test
    
    def create_nbow_features(self, word_vectors: Dict[str, np.ndarray], 
                            label: str) -> Tuple[np.ndarray, np.ndarray]:
        """创建NBOW特征（词向量求和）"""
        from src.serving.packages.text_regularization import extractWords
        
        filepath = os.path.join(self.data_dir, f'msg{label}.log.seg')
        embeddings = []
        
        with open(filepath, encoding='utf-8') as f:
            for line in f:
                line = extractWords(line)
                words = line.strip().split(' ')
                text_embedding = np.zeros(200)
                for word in words:
                    try:
                        text_embedding += word_vectors[word]
                    except KeyError:
                        text_embedding += word_vectors.get('UNK', np.zeros(200))
                embeddings.append(text_embedding)
        
        embeddings = np.asarray(embeddings, dtype=np.float32)
        labels = np.zeros(len(embeddings)) if label == 'spam' else np.ones(len(embeddings))
        
        return embeddings, labels
    
    def create_sequence_features(self, word_vectors: Dict[str, np.ndarray], 
                                label: str) -> Tuple[list, np.ndarray]:
        """创建序列特征（词向量序列）"""
        from src.serving.packages.text_regularization import extractWords
        
        filepath = os.path.join(self.data_dir, f'msg{label}.log.seg')
        samples = []
        
        with open(filepath, encoding='utf-8') as f:
            for line in f:
                line = extractWords(line)
                words = line.strip().split(' ')
                text_sequence = []
                for word in words:
                    try:
                        text_sequence.append(word_vectors[word])
                    except KeyError:
                        text_sequence.append(word_vectors.get('UNK', np.zeros(200)))
                samples.append(text_sequence)
        
        labels = np.zeros(len(samples)) if label == 'spam' else np.ones(len(samples))
        
        return samples, labels
