"""服务部署模块"""
import logging
import os
import numpy as np
import tensorflow as tf
from typing import Dict, Any

from src.config import Config


class TextClassifierService:
    """文本分类服务"""

    def __init__(self, model_type: str = 'rnn'):
        self.model_type = model_type
        self.word_vectors = None
        self.model = None
        self._load_resources()

    def _load_resources(self):
        """加载词向量和模型"""
        from src.utils import load_embedding
        from src.classifier import RNNClassifier, MLPClassifier, CNNClassifier

        logging.info("加载词向量...")
        self.word_vectors = load_embedding(os.path.join(Config.WEIGHTS_DIR, 'model_word2vec_200.npy'))

        logging.info(f"加载{self.model_type}分类器...")
        if self.model_type == 'rnn':
            self.model = RNNClassifier(
                lstm_units=Config.RNN_CONFIG['lstm_units'],
                recurrent_dropout=Config.RNN_CONFIG['recurrent_dropout']
            )
            self.model.load(os.path.join(Config.WEIGHTS_DIR, 'rnn_classifier'))
        elif self.model_type == 'mlp':
            self.model = MLPClassifier(
                hidden_units=Config.MLP_CONFIG['hidden_units'],
                dropout_keep=Config.MLP_CONFIG['dropout_keep']
            )
            self.model.load(os.path.join(Config.WEIGHTS_DIR, 'mlp_classifier'))
        elif self.model_type == 'cnn':
            self.model = CNNClassifier(
                n_filter=Config.CNN_CONFIG['n_filter'],
                filter_size=Config.CNN_CONFIG['filter_size'],
                stride=Config.CNN_CONFIG['stride'],
                pool_size=Config.CNN_CONFIG['pool_size'],
                pool_strides=Config.CNN_CONFIG['pool_strides'],
                dropout_keep=Config.CNN_CONFIG['dropout_keep']
            )
            self.model.load(os.path.join(Config.WEIGHTS_DIR, 'cnn_classifier'))

    def preprocess(self, text: str) -> np.ndarray:
        """预处理文本"""
        from src.text_features import extractWords

        text = extractWords(text)
        words = text.strip().split(' ')

        if self.model_type == 'mlp':
            embedding = np.zeros(200)
            for word in words:
                embedding += self.word_vectors.get(word, self.word_vectors.get('UNK', np.zeros(200)))
            return embedding.reshape(1, -1)
        else:
            sequence = []
            max_seq_len = 20  # RNN和CNN都使用20的序列长度

            for word in words[:max_seq_len]:
                sequence.append(self.word_vectors.get(word, self.word_vectors.get('UNK', np.zeros(200))))

            while len(sequence) < max_seq_len:
                sequence.append(np.zeros(200))

            return np.array(sequence).reshape(1, max_seq_len, 200)

    def predict(self, text: str) -> Dict[str, Any]:
        """预测文本分类"""
        input_data = self.preprocess(text)
        input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)

        logits = self.model(input_data, training=False)
        prob = tf.nn.softmax(logits)
        pred_label = tf.argmax(prob, axis=1).numpy()[0]
        confidence = prob.numpy()[0][pred_label]

        return {
            'text': text,
            'label': 'normal' if pred_label == 1 else 'spam',
            'confidence': float(confidence),
            'model_type': self.model_type
        }

    def batch_predict(self, texts: list) -> list:
        """批量预测"""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results


class TFSService:
    """TensorFlow Serving客户端"""

    def __init__(self, host: str = 'localhost', port: int = 8501):
        self.host = host
        self.port = port
        self.predict_fn = None
        self._init_client()

    def _init_client(self):
        """初始化TF Serving客户端"""
        import json
        import requests

        self.predict_url = f'http://{self.host}:{self.port}/v1/models/text_classifier:predict'

        def predict(text: str) -> Dict[str, Any]:
            from src.text_features import extractWords
            from src.utils import load_embedding

            word_vectors = load_embedding(os.path.join(Config.WEIGHTS_DIR, 'model_word2vec_200.npy'))

            text = extractWords(text)
            words = text.strip().split(' ')
            sequence = []

            for word in words[:20]:
                sequence.append(word_vectors.get(word, word_vectors.get('UNK', np.zeros(200))))

            while len(sequence) < 20:
                sequence.append(np.zeros(200))

            input_data = np.array(sequence).reshape(1, 20, 200)

            request_data = json.dumps({
                'signature_name': 'serving_default',
                'instances': input_data.tolist()
            })

            response = requests.post(self.predict_url, data=request_data)
            result = response.json()

            pred_label = int(np.argmax(result['predictions'][0]))
            confidence = float(result['predictions'][0][pred_label])

            return {
                'text': text,
                'label': 'normal' if pred_label == 1 else 'spam',
                'confidence': confidence
            }

        self.predict_fn = predict

    def predict(self, text: str) -> Dict[str, Any]:
        """通过TF Serving预测"""
        return self.predict_fn(text)


def export_model_for_serving(model_type: str = 'rnn', export_dir: str = None):
    """导出模型用于TensorFlow Serving"""
    if export_dir is None:
        export_dir = os.path.join(Config.OUTPUT_DIR, 'saved_model', model_type)

    from src.classifier import RNNClassifier, MLPClassifier, CNNClassifier

    logging.info(f"导出{model_type}模型...")

    if model_type == 'rnn':
        model = RNNClassifier()
        model.load(os.path.join(Config.WEIGHTS_DIR, 'rnn_classifier'))
    elif model_type == 'mlp':
        model = MLPClassifier()
        model.load(os.path.join(Config.WEIGHTS_DIR, 'mlp_classifier'))
    elif model_type == 'cnn':
        model = CNNClassifier()
        model.load(os.path.join(Config.WEIGHTS_DIR, 'cnn_classifier'))

    tf.saved_model.save(model, export_dir)

    logging.info(f"模型已导出到 {export_dir}")


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.utils import setup_logging

    setup_logging()
    Config.ensure_dirs()

    service = TextClassifierService(model_type='rnn')
    test_texts = [
        '免费领取手机话费充值卡',
        '今天天气不错',
        '恭喜您中了大奖，请点击链接领取'
    ]

    for text in test_texts:
        result = service.predict(text)
        logging.info(f"文本: {text}")
        logging.info(f"预测结果: {result['label']}, 置信度: {result['confidence']:.4f}")