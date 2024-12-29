import time
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def timeit(func):
    """関数の実行時間を計測するデコレーター"""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(
            f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds"
        )
        return result

    return wrapper
