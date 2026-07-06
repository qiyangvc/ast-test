from .config import Config

__all__ = ['Config', 'setup_logging', 'accuracy', 'DataLoader']


def __getattr__(name):
    """Lazy imports keep data-preparation utilities free of training deps."""
    if name in {'setup_logging', 'accuracy'}:
        from .utils import setup_logging, accuracy
        return {'setup_logging': setup_logging, 'accuracy': accuracy}[name]
    if name == 'DataLoader':
        from .data_loader import DataLoader
        return DataLoader
    raise AttributeError(f"module 'src' has no attribute {name!r}")
