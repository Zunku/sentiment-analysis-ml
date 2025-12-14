
import nltk
nltk.download('all')
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import pandas as pd
import mlflow
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from custom_transformers import import_data, DropNullData, DropDuplicates
from text_utils import re_breakline, re_dates, re_hiperlinks, re_money, re_negation, re_numbers, \
    re_special_chars, re_whitespaces, ApplyRegex, StemmingProcess, StopWordsRemoval
from ml_utils import BinaryClassifiersAnalysis, cross_val_performance


parser = argparse.ArgumentParser()
parser.add_argument('--training_data', type=str)
args = parser.parse_args()

orders_full_imputed = pd.read_csv(args.training_data)

X = orders_full_imputed.drop(['review_score','product_category','payment_type'], axis=1)
y = orders_full_imputed.review_score

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, train_size=.8, test_size=.2)


#with mlflow.start_run():
 #   model = LogisticRegression(C=10, penalty='l1',class_weight='balanced',random_state=42,solver='liblinear')
  #  model.fit(X_train, y_train)
   # y_pred = model.predict(X_test)
#
 #   accuracy = accuracy_score(y_test,y_pred)
  #  mlflow.log_metric('accuracy',accuracy)