"""文本正则化处理模块"""
import re
import jieba


def extractWords(text: str) -> str:
    """提取文本中的中文词语"""
    # 去除特殊字符和标点
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    
    # 使用结巴分词
    words = jieba.cut(text)
    
    # 过滤单字，保留词语
    result = [word for word in words if len(word) > 1]
    
    return ' '.join(result)


def clean_text(text: str) -> str:
    """清洗文本"""
    # 去除HTML标签
    text = re.sub(r'<.*?>', '', text)
    
    # 去除URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def preprocess(text: str) -> str:
    """完整预处理流程"""
    text = clean_text(text)
    text = extractWords(text)
    return text
