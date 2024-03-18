import pandas as pd
import logging


def get_charset(df: pd.DataFrame) -> str:
    # from https://github.com/NastyBoget/hrtr -- Attention Model
    char_set = set()
    for _, row in df.iterrows():
        char_set = char_set | set(str(row["text"]))
    return "".join(sorted(list(char_set)))

def get_logger(out_file: str) -> logging.Logger:
    # from https://github.com/NastyBoget/hrtr -- Attention Model
    logging.basicConfig(filename=out_file,
                        filemode="a",
                        level=logging.INFO,
                        format="%(message)s")

    logger = logging.getLogger('python-logstash-logger')
    logger.setLevel(logging.INFO)
    return logger
