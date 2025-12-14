
# import libraries
import mlflow
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from scipy.sparse import load_npz
import os
#from mlflow.models.signature import infer_signature

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

def main(args):
    with mlflow.start_run():
        # read data
        X_prep, y = get_data_from_folder(args.training_data)

        # split data
        X_train, X_test, y_train, y_test = split_data(X_prep, y)

        # train model
        class_weight = None if args.class_weight == 'Nini' else args.class_weight
        model = train_model(class_weight, args.penalty, args.reg_rate, X_train, X_test, y_train, y_test)

        ########## NUEVOOO
        # create the signature by inferring it from the datasets
        #signature = infer_signature(X_prep, y)

        # manually log the model
        #mlflow.sklearn.log_model(model, artifact_path='model_output', signature=signature)

        # evaluate model
        eval_model(model, X_train, y_train, X_test, y_test)

# function that reads the data
def get_data_from_folder(folder_path):
    print("Reading data...")

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.startswith('comments'):
            X = pd.read_csv(file_path)
        elif file_name.startswith('target'):
            y = pd.read_csv(file_path)
        elif file_name.endswith('.npz'):
            X_prep = load_npz(file_path)

    return X_prep, y

# function that splits the data
def split_data(X_prep, y):
    print("Splitting data...")
    y = pd.Series(y['target'].values)
    X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=0.20, random_state=0)
    return X_train, X_test, y_train, y_test

# function that trains the model
def train_model(class_weight, penalty, reg_rate, X_train, X_test, y_train, y_test):
    mlflow.log_param("Regularization rate", reg_rate)
    mlflow.log_param('Class Weight', class_weight)
    mlflow.log_param('Penalty', penalty)
    print("Training model...")
    
    # Training model
    model = LogisticRegression(C=reg_rate, solver="liblinear", penalty=penalty, class_weight=class_weight).fit(X_train, y_train)

    # Log with MLFlow to compare metrics and easy registry
    mlflow.sklearn.log_model(model, artifact_path='model_output')

    # Localy save model to pass as an output to the next component in the pipeline
    mlflow.sklearn.save_model(model, path=args.model_output)
    return model

# function that evaluates the model
def eval_model(model, X_test, y_test, X_train, y_train):
    print('Evaluating model...')
    # Calculate performance
    # performance = trainer.evaluate_performance(X_train, y_train, X_test, y_test, cv=5, save=False)
    
    # Calculate accuracy
    y_pred = model.predict(X_test)
    acc = np.average(y_pred == y_test)
    print('Accuracy:', acc)
    mlflow.log_metric("Accuracy", acc)

    # calculate AUC
    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test,y_scores[:,1])
    print('AUC: ' + str(auc))
    mlflow.log_metric("AUC", auc)

    # Calculate F1 Score
    f1 = f1_score(y_test, y_pred)
    mlflow.log_metric('f1_score', f1)

    # plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
    fig = plt.figure(figsize=(6, 4))
    # Plot the diagonal 50% line
    plt.plot([0, 1], [0, 1], 'k--')
    # Plot the FPR and TPR achieved by our model
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig("ROC-Curve.png")
    mlflow.log_artifact("ROC-Curve.png")    

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--reg_rate", dest='reg_rate',
                        type=float)
    parser.add_argument('--class_weight', dest='class_weight',
                        type=str)
    parser.add_argument('--penalty', dest='penalty',
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
