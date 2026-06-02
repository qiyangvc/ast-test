"""特征提取模块"""
import logging
import os
import re
import jieba
import numpy as np
from typing import Dict, Tuple

from src.config import Config


def extractWords(text: str) -> str:
    """提取文本中的中文词语"""
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    words = jieba.cut(text)
    result = [word for word in words if len(word) > 1]
    return ' '.join(result)


class FeatureExtractor:
    """特征提取器基类"""
    
    def __init__(self, word_vectors: Dict[str, np.ndarray]):
        self.word_vectors = word_vectors
    
    def extract(self, text: str) -> np.ndarray:
        """提取特征"""
        raise NotImplementedError


class NBOWFeatureExtractor(FeatureExtractor):
    """NBOW（词袋词向量）特征提取器"""
    
    def __init__(self, word_vectors: Dict[str, np.ndarray]):
        super().__init__(word_vectors)
    
    def extract(self, text: str) -> np.ndarray:
        """提取NBOW特征（词向量求和）"""
        text = extractWords(text)
        words = text.strip().split(' ')
        embedding = np.zeros(200)
        
        for word in words:
            embedding += self.word_vectors.get(word, self.word_vectors.get('UNK', np.zeros(200)))
        
        return embedding
    
    def process_file(self, filepath: str, label: int) -> Tuple[np.ndarray, np.ndarray]:
        """处理文件生成特征"""
        features = []
        labels = []
        
        with open(filepath, encoding='utf-8') as f:
            for line in f:
                features.append(self.extract(line))
                labels.append(label)
        
        return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)


class SequenceFeatureExtractor(FeatureExtractor):
    """序列特征提取器（用于RNN/CNN）"""
    
    def __init__(self, word_vectors: Dict[str, np.ndarray], max_seq_len: int = 20):
        super().__init__(word_vectors)
        self.max_seq_len = max_seq_len
    
    def extract(self, text: str) -> np.ndarray:
        """提取序列特征"""
        text = extractWords(text)
        words = text.strip().split(' ')
        sequence = []
        
        for word in words[:self.max_seq_len]:
            sequence.append(self.word_vectors.get(word, self.word_vectors.get('UNK', np.zeros(200))))
        
        # Padding
        while len(sequence) < self.max_seq_len:
            sequence.append(np.zeros(200))
        
        return np.array(sequence, dtype=np.float32)
    
    def process_file(self, filepath: str, label: int) -> Tuple[np.ndarray, np.ndarray]:
        """处理文件生成特征"""
        features = []
        labels = []
        
        with open(filepath, encoding='utf-8') as f:
            for line in f:
                features.append(self.extract(line))
                labels.append(label)
        
        return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)


def prepare_mlp_features():
    """准备MLP分类器特征"""
    from src.utils import load_embedding
    
    logging.info("加载词向量...")
    word_vectors = load_embedding(os.path.join(Config.WEIGHTS_DIR, 'model_word2vec_200.npy'))
    
    extractor = NBOWFeatureExtractor(word_vectors)
    
    logging.info("处理垃圾短信...")
    spam_path = os.path.join(Config.MSG_LOG_DIR, 'msgspam.log.seg')
    spam_x, spam_y = extractor.process_file(spam_path, 0)
    
    logging.info("处理正常短信...")
    pass_path = os.path.join(Config.MSG_LOG_DIR, 'msgpass.log.seg')
    pass_x, pass_y = extractor.process_file(pass_path, 1)
    
    # 合并数据
    x = np.concatenate([spam_x, pass_x], axis=0)
    y = np.concatenate([spam_y, pass_y], axis=0)
    
    # 保存
    output_path = os.path.join(Config.OUTPUT_DIR, 'features_mlp.npz')
    np.savez(output_path, x=x, y=y)
    
    logging.info(f"MLP特征已保存到 {output_path}")


def prepare_sequence_features(max_seq_len: int = 20):
    """准备序列特征（用于RNN/CNN）"""
    from src.utils import load_embedding
    
    logging.info("加载词向量...")
    word_vectors = load_embedding(os.path.join(Config.WEIGHTS_DIR, 'model_word2vec_200.npy'))
    
    extractor = SequenceFeatureExtractor(word_vectors, max_seq_len)
    
    logging.info("处理垃圾短信...")
    spam_path = os.path.join(Config.MSG_LOG_DIR, 'msgspam.log.seg')
    spam_x, spam_y = extractor.process_file(spam_path, 0)
    
    logging.info("处理正常短信...")
    pass_path = os.path.join(Config.MSG_LOG_DIR, 'msgpass.log.seg')
    pass_x, pass_y = extractor.process_file(pass_path, 1)
    
    # 合并数据
    x = np.concatenate([spam_x, pass_x], axis=0)
    y = np.concatenate([spam_y, pass_y], axis=0)
    
    # 保存
    output_path = os.path.join(Config.OUTPUT_DIR, f'features_seq_{max_seq_len}.npz')
    np.savez(output_path, x=x, y=y)
    
    logging.info(f"序列特征已保存到 {output_path}")


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.utils import setup_logging
    
    setup_logging()
    Config.ensure_dirs()
    
    # 准备MLP特征
    prepare_mlp_features()
    
    # 准备序列特征
    prepare_sequence_features()
