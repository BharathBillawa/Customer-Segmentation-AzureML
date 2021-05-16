import pandas as pd
import numpy as np
import argparse
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

KEY_GENDER = 'Gender'
KEY_MARRIED = 'Ever_Married'
KEY_GRADUATED = 'Graduated'
KEY_PROFESSION = 'Profession'
KEY_SPENDING_SCORE = 'Spending_Score'
KEY_VAR = 'Var_1'
KEY_SEGMENTATION = 'Segmentation'
KEY_TARGET_FEATURE = KEY_SEGMENTATION

VALUE_MALE = 'Male'
VALUE_FEMALE = 'Female'
VALUE_NO = 'No'
VALUE_YES = 'Yes'

CORR_THRESH = 0.01
TEST_DATA_SPLIT = 0.2

class CustSegModel():
  """
  Holder class for model and utilities.
  """
  def __init__(self, cat_data_encoder, data_mode, low_corr_labels, model):
    self.cat_data_encoder = cat_data_encoder
    self.data_mode = data_mode
    self.low_corr_labels = low_corr_labels
    self.model = model


class CatDataEncoder():
  """
  Holder class for the categorical data encoders.
  """
  def __init__(self):
    self.profession_encoder = LabelEncoder()
    self.spending_score_encoder = LabelEncoder()
    self.var_encoder = LabelEncoder()
    self.segment_encoder = LabelEncoder()

def impute_na(data):
  """
  Imputes missing data with mode.
  Returns: mode, imputed data
  """
  data_mode = data.mode().iloc[0, :]
  return data_mode, data.fillna(data_mode)

def encode_cat_data(data):
  """
  Encodes categorical params in `data`.
  Returns: data encoder, encoded data
  """
  # Encode binary params.
  data[KEY_GENDER] = data[KEY_GENDER].replace({VALUE_MALE: 0, VALUE_FEMALE: 1})
  data[KEY_MARRIED] = data[KEY_MARRIED].replace({VALUE_NO: 0, VALUE_YES: 1})
  data[KEY_GRADUATED] = data[KEY_GRADUATED].replace({VALUE_NO: 0, VALUE_YES: 1})

  # Encode non-binary params.
  cat_data_encoder = CatDataEncoder()
  data[KEY_PROFESSION] = (
    cat_data_encoder
      .profession_encoder
      .fit_transform(
        data[KEY_PROFESSION]
      )
  )
  data[KEY_SPENDING_SCORE] = (
    cat_data_encoder
      .spending_score_encoder
      .fit_transform(
        data[KEY_SPENDING_SCORE]
      )
  )
  data[KEY_VAR] = (
    cat_data_encoder
      .var_encoder
      .fit_transform(
        data[KEY_VAR]
      )
  )
  data[KEY_SEGMENTATION] = (
    cat_data_encoder
      .segment_encoder
      .fit_transform(
        data[KEY_SEGMENTATION]
      )
  )

  return cat_data_encoder, data

def drop_low_corr_features(data, corr_thresh = CORR_THRESH):
  """
  Drops features with low correlation.
  Returns: Feature labels with low correlation, filtered data
  """
  feature_corr = data.corr()
  target_corr = feature_corr[KEY_TARGET_FEATURE]
  low_corr_labels = target_corr[abs(target_corr) < corr_thresh].index
  filtered_data = data.drop(low_corr_labels, axis = 1)

  return low_corr_labels, filtered_data

def search_classifiers(X_train, X_test, y_train, y_test):
  """
  Searches and returns model with best accuracy.
  """
  names = [
    "Nearest Neighbors", "Linear SVM", "RBF SVM",
    "Decision Tree", "Random Forest", "Neural Net",
    "AdaBoost", "Naive Bayes", "QDA"
  ]

  classifiers = [
      KNeighborsClassifier(3),
      SVC(kernel = "linear", C = 0.025),
      SVC(gamma = 2, C = 1),
      DecisionTreeClassifier(max_depth = 5),
      RandomForestClassifier(max_depth = 5, n_estimators = 10, max_features = 1),
      MLPClassifier(alpha = 1, hidden_layer_sizes = (5, 3), max_iter = 1000),
      AdaBoostClassifier(),
      GaussianNB(),
      QuadraticDiscriminantAnalysis()
  ]

  print('Accuracies')
  best_accuracy = 0
  best_model = None
  for name, clf in zip(names, classifiers):
      clf.fit(X_train, np.ravel(y_train))
      y_pred = clf.predict(X_test)
      accuracy = metrics.accuracy_score(y_test, y_pred)
      print(name, ': ', accuracy)

      if accuracy > best_accuracy:
        best_model = clf

  return best_model

def save_model(cust_seg_model):
  """
  Saves the model in 'cust-seg-model.pkl' file.
  """
  joblib.dump(cust_seg_model, 'cust_seg_model.joblib')
  ## OR
  # model_file = open('cust_seg_model.pkl', 'wb')
  # pickle.dump(cust_seg_model, model_file)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--data_path',
    type = str,
    help = 'Path to the training data'
  )
  
  args = parser.parse_args()
  
  train_df = pd.read_csv(args.data_path + '/train.csv')

  # Impute missing data with mode
  data_mode, train_df = impute_na(train_df)

  # Encode categorical data
  cat_data_encoder, train_df = encode_cat_data(train_df)

  # Drop low correlating features
  low_corr_labels, filtered_data = drop_low_corr_features(train_df)

  # Split dataset for training and testing
  X_train, X_test, y_train, y_test = train_test_split(
    filtered_data.iloc[:,:-1],
    filtered_data.iloc[:, -1:],
    test_size = TEST_DATA_SPLIT
  )

  # Search and print classifier accuracies
  model = search_classifiers(X_train, X_test, y_train, y_test)
  
  cust_seg_model = CustSegModel(cat_data_encoder, data_mode, low_corr_labels, model)
  save_model(cust_seg_model)

