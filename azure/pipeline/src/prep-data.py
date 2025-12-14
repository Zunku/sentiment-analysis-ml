import nltk
nltk.download('all')
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import save_npz

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utilities.custom_transformers import import_data, DropNullData, DropDuplicates
from utilities.text_utils import re_breakline, re_dates, re_hiperlinks, re_money, re_negation, re_numbers, \
    re_special_chars, re_whitespaces, ApplyRegex, StemmingProcess, StopWordsRemoval
from utilities.text_prep import text_transformers
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

"""
-----------------------------------
------ 0. PROJECT VARIABLES -------
-----------------------------------
"""

# Variables for reading the data
FILENAME = 'olist_order_reviews_dataset.csv'
COLS_READ = ['review_comment_message', 'review_score']
CORPUS_COL = 'review_comment_message'
TARGET_COL = 'target'

# Defining stopwords
PT_STOPWORDS = stopwords.words('portuguese')

"""
This python script will allocate all the custom transformers that are specific for the project task.
The idea is to encapsulate the classes and functions used on pipelines to make codes cleaner.

"""

"""
-----------------------------------
----- 1. CUSTOM TRANSFORMERS ------
           1.1 Classes
-----------------------------------
"""


class ColumnMapping(BaseEstimator, TransformerMixin):
    """
    This class applies the map() function into a DataFrame for transforming a columns given a mapping dictionary

    Parameters
    ----------
    :param old_col_name: name of the columns where mapping will be applied [type: string]
    :param mapping_dict: python dictionary with key/value mapping [type: dict]
    :param new_col_name: name of the new column resulted by mapping [type: string, default: 'target]
    :param drop: flag that guides the dropping of the old_target_name column [type: bool, default: True]

    Returns
    -------
    :return X: pandas DataFrame object after mapping application [type: pd.DataFrame]

    Application
    -----------
    # Transforming a DataFrame column given a mapping dictionary
    mapper = ColumnMapping(old_col_name='col_1', mapping_dict=dictionary, new_col_name='col_2', drop=True)
    df_mapped = mapper.fit_transform(df)
    """

    def __init__(self, old_col_name, mapping_dict, new_col_name='target', drop=True):
        self.old_col_name = old_col_name
        self.mapping_dict = mapping_dict
        self.new_col_name = new_col_name
        self.drop = drop

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Applying mapping
        X[self.new_col_name] = X[self.old_col_name].map(self.mapping_dict)

        # Dropping the old columns (if applicable)
        if self.drop:
            X.drop(self.old_col_name, axis=1, inplace=True)

        return X

def main(args):
    # Read data
    df = get_data(args.input_data)

    X, X_prep, y = clean_data(df)

    # Crea la carpeta si no existe
    output_folder = args.output_folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    save_npz(Path(output_folder) / 'vectorized_comments.npz', X_prep)
    # X_prep.tocsc((Path(output_folder) / 'vectorized_comments.csc'))
    y.to_csv(Path(output_folder) / 'target.csv')
    X.to_csv(Path(output_folder) / 'comments.csv')

# Function that reads data
def get_data(path):
    df = pd.read_csv(path)

    row_count = (len(df))
    print(f'Preparing {row_count} rows of data'.format(row_count))
    return df

# Funtion that removes missing values, drop duplicates, apply text transformers, and maps the target (review_score) to positive/negative reviews
def clean_data(df):
    # Creating a dictionary for mapping the target column based on review score
    score_map = {
        1: 0,
        2: 0,
        3: 0,
        4: 1,
        5: 1
    }

    # Selecting columns
    df_prep = df[COLS_READ]
    # Creating a pipeline for the initial prep on the data
    initial_prep_pipeline = Pipeline([
        ('mapper', ColumnMapping(old_col_name='review_score', mapping_dict=score_map, new_col_name=TARGET_COL)),
        ('null_dropper', DropNullData()),
        ('dup_dropper', DropDuplicates())
    ])
    # Applying initial prep pipeline
    df_prep = initial_prep_pipeline.fit_transform(df_prep)

    # Applying text transformations
    X = df_prep[CORPUS_COL]
    text_list = df_prep[CORPUS_COL].to_list()
    text_pipeline = text_transformers()
    X_prep = text_pipeline.fit_transform(text_list)
    y = df_prep[TARGET_COL]
    print(X)
    return X, X_prep, y

    # Setup arg parser
def parse_args():
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument('--input_data', dest='input_data', type=str)
    parser.add_argument('--output_folder', dest='output_folder', type=str)

    args = parser.parse_args()
    return args

# Run script
if __name__ == '__main__':
    # Add space in logs
    print('\n\n')
    print('*' * 60)

    # Parse args
    args = parse_args()

    # Run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
