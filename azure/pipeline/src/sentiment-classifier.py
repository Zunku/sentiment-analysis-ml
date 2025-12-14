
# import libraries
import nltk
nltk.download('all')
import mlflow
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from scipy.sparse import load_npz
import os
from sklearn.model_selection import GridSearchCV
from mlflow.models.signature import infer_signature
import mlflow.pyfunc

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utilities.ml_utils import BinaryClassifiersAnalysis, cross_val_performance
from utilities.text_prep import text_transformers

def main(args):
    with mlflow.start_run():
        # read data
        X, X_prep, y, model = get_data_and_model(args.training_data, args.model_input)

        # train sentiment classifier
        sentiment_classifier = train_sentiment_classifier(X, X_prep, y, model)


def get_data_and_model(folder_path, model_path):
    print("Reading data...")

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.startswith('comments'):
            X = pd.read_csv(file_path)
        elif file_name.startswith('target'):
            y = pd.read_csv(file_path)
        elif file_name.endswith('.npz'):
            X_prep = load_npz(file_path)

    model = mlflow.sklearn.load_model(model_path)
    print(f'Loaded model from: {model_path}')

    return X, X_prep, y, model

def train_sentiment_classifier(X, X_prep, y, model):

    # Creating a complete pipeline for prep and predict
    text_pipeline = text_transformers()
    e2e_pipeline = Pipeline([
        ('text_prep', text_pipeline),
        ('model', model)
        ])
    print('Created pipeline')

    # Defining a param grid for searching best pipelines options
    """
    param_grid = [{
        'text_prep__vectorizer__max_features': np.arange(500, 851, 50),
        'text_prep__vectorizer__min_df': [7, 9, 12, 15, 30],
        'text_prep__vectorizer__max_df': [.4, .5, .6, .7]
    }]
    """

    param_grid = [{
    'text_prep__vectorizer__max_features': np.arange(500, 501, 50),
    'text_prep__vectorizer__min_df': [7],
    'text_prep__vectorizer__max_df': [.4]
    }]

    # Searching for the best options
    grid_search_prep = GridSearchCV(e2e_pipeline, param_grid, cv=5, scoring='f1', verbose=1, n_jobs=-1)
    text_list = X['review_comment_message'].to_list()
    y = pd.Series(y['target'].values)
    grid_search_prep.fit(text_list,y)
    print('Best params after a complete search: ')
    print(grid_search_prep.best_params_)

    # Returning the best options
    vectorizer_max_features = grid_search_prep.best_params_['text_prep__vectorizer__max_features']
    vectorizer_min_df = grid_search_prep.best_params_['text_prep__vectorizer__min_df']
    vectorizer_max_df = grid_search_prep.best_params_['text_prep__vectorizer__max_df']

    # Updating the e2e pipeline with the best options found on search
    e2e_pipeline.named_steps['text_prep'].named_steps['vectorizer'].max_features = vectorizer_max_features
    e2e_pipeline.named_steps['text_prep'].named_steps['vectorizer'].min_df = vectorizer_min_df
    e2e_pipeline.named_steps['text_prep'].named_steps['vectorizer'].max_df = vectorizer_max_df

    # Fitting the model again
    e2e_pipeline.fit(text_list, y)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    # Retrieving performance for te final model after hyperparam updating
    final_model = e2e_pipeline.named_steps['model']
    final_performance = cross_val_performance(final_model, X_prep, y, cv=5)
    print(final_performance)

    # Infer signature from the data
    df = pd.DataFrame({
    'review_comment_message': ["text1", "text2", "text3"]
    })
    signature = infer_signature(df, e2e_pipeline.predict(text_list))

    # Custom environment
    conda_env = {
        "channels": ["conda-forge"], 
        "dependencies": [ 
            "python=3.10", 
            "pip==24.0", 
            {"pip": [ 
                "mlflow", 
                "numpy==1.23.5", 
                "pandas", 
                'cloudpickle', 
                'nltk', 
                'scikit-learn']} 
                ],
                "name": "mlflow-py311"
                }

    # Wrap the model to use as a PythonModel
    class TextPipelineWrapper(mlflow.pyfunc.PythonModel):
        def __init__(self, pipeline):
            self.pipeline = pipeline

        def predict(self, context, model_input):
            # model_input llega como DataFrame o lista de textos
            if isinstance(model_input, pd.DataFrame):
                texts = model_input.iloc[:, 0].astype(str).tolist()
            elif isinstance(model_input, list):
                texts = [str(t) for t in model_input]
            else:
                texts = [str(model_input)]
            
            return self.pipeline.predict(texts)
    

    # Log with MLFlow to easy registry using pyfunc
    # wrapped_model = TextPipelineWrapper(e2e_pipeline)
    # mlflow.pyfunc.log_model(
    #     artifact_path="model_output",
    #     python_model=wrapped_model,
    #     signature=signature,
    #     conda_env=conda_env
    # )

    # mlflow.pyfunc.save_model(
    # python_model=wrapped_model, 
    # path=args.model_output, 
    # signature=signature,
    # conda_env=conda_env
    # )
    
    # Log and save with MLFlow sklearn
    mlflow.sklearn.log_model(e2e_pipeline, artifact_path='model_output', signature=signature, conda_env=conda_env)
    mlflow.sklearn.save_model(e2e_pipeline, path=args.model_output, signature=signature, conda_env=conda_env)
    return e2e_pipeline

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument('--model_input', dest='model_input', 
                        type=str)
    parser.add_argument('--model_output', dest='model_output',
                        type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
