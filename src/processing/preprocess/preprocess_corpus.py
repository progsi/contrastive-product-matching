import pandas as pd
import numpy as np
np.random.seed(42)
import random
random.seed(42)
import os

from src.data import utils


def load_json_lines_to_dataframe(file_path):
    """
    Load JSON lines from a file and convert them to a pandas DataFrame.

    Parameters:
    - file_path (str): The path to the JSON lines file.

    Returns:
    - pd.DataFrame: The resulting DataFrame.
    """
    import json 

    data = []

    with open(file_path, 'r') as file:
        for line in file:
            try:
                json_obj = json.loads(line)
                data.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {e}")

    df = pd.DataFrame(data)
    return df


if __name__ == '__main__':

    print('PREPROCESSING CORPUS')

    # FIXME: why doesnt this work?
    # corpus = pd.read_json('data/raw/wdc-lspc/corpus/offers_corpus_english_v2_non_norm.json', lines=True)
    corpus = load_json_lines_to_dataframe('data/raw/wdc-lspc/corpus/offers_corpus_english_v2_non_norm.json.gz')
    
    # preprocess english corpus

    print('BUILDING PREPROCESSED CORPUS...')
    corpus['title'] = corpus['title'].apply(utils.clean_string_wdcv2)
    corpus['description'] = corpus['description'].apply(utils.clean_string_wdcv2)
    corpus['brand'] = corpus['brand'].apply(utils.clean_string_wdcv2)
    corpus['price'] = corpus['price'].apply(utils.clean_string_wdcv2)
    corpus['specTableContent'] = corpus['specTableContent'].apply(utils.clean_specTableContent_wdcv2)

    os.makedirs(os.path.dirname('data/interim/wdc-lspc/corpus/'), exist_ok=True)
    corpus.to_pickle('data/interim/wdc-lspc/corpus/preprocessed_english_corpus.pkl.gz')
    print('FINISHED BUILDING PREPROCESSED CORPUS...')
