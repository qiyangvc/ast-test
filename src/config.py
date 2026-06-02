import os
from typing import Dict, Any

class Config:
    """项目配置类"""
    
    # 项目根目录
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 数据目录
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MSG_LOG_DIR = os.path.join(DATA_DIR, 'msglog')
    
    # 输出目录
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
    WEIGHTS_DIR = os.path.join(BASE_DIR, 'weights')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    
    # Word2Vec 配置
    WORD2VEC_CONFIG = {
        'min_freq': 5,
        'batch_size': 128,
        'embedding_size': 200,
        'skip_window': 2,
        'num_skips': 4,
        'num_sampled': 64,
        'learning_rate': 0.001,
        'n_epoch': 5,
        'print_freq': 100,
        'model_name': 'model_word2vec_200'
    }
    
    # 分类器通用配置
    CLASSIFIER_CONFIG = {
        'test_size': 0.2,
        'batch_size': 4,
        'display_step': 1
    }
    
    # RNN分类器配置
    RNN_CONFIG = {
        'learning_rate': 0.001,
        'n_epoch': 5,
        'lstm_units': 32,
        'recurrent_dropout': 0.2
    }
    
    # MLP分类器配置
    MLP_CONFIG = {
        'learning_rate': 0.01,
        'n_epoch': 5,
        'hidden_units': 50,
        'dropout_keep': 0.5
    }
    
    # CNN分类器配置
    CNN_CONFIG = {
        'learning_rate': 0.005,
        'n_epoch': 5,
        'max_seq_len': 10,
        'n_filter': 4,
        'filter_size': 3,
        'stride': 1,
        'pool_size': 2,
        'pool_strides': 1,
        'dropout_keep': 0.5
    }
    
    @classmethod
    def get(cls, key: str) -> Any:
        """获取配置值"""
        return getattr(cls, key, None)
    
    @classmethod
    def ensure_dirs(cls):
        """确保所有必要目录存在"""
        dirs = [
            cls.DATA_DIR,
            cls.MSG_LOG_DIR,
            cls.OUTPUT_DIR,
            cls.WEIGHTS_DIR,
            cls.LOGS_DIR
        ]
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
