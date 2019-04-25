# GCT634 (2018) HW1
#
# Mar-18-2018: initial version
#
# Juhan Nam
#

import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

from feature_summary import *
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import VotingClassifier

FEATURE_NAMES = ([f'mfcc0_{i:02}' for i in range(13)]
                 + [f'mfcc1_{i:02}' for i in range(13)]
                 + [f'mfcc2_{i:02}' for i in range(13)]
                 + [f'mfcc3_{i:02}' for i in range(13)]

                 + [f'spec0_{i:02}' for i in range(40)]
                 + [f'spec1_{i:02}' for i in range(40)]
                 + [f'spec2_{i:02}' for i in range(40)]
                 + [f'spec3_{i:02}' for i in range(40)]

                 + ['average_loudness', 'dissonance', 'dynamic_complexity',
                    'pitch_salience', 'silence_rate_20dB', 'silence_rate_30dB',
                    'silence_rate_60dB', 'spectral_centroid', 'spectral_complexity', 'zerocrossingrate'])
FEATURE_NAMES = np.array(FEATURE_NAMES)


def train_model(train_X, train_Y, valid_X, valid_Y, hyper_param1):
  # Choose a classifier (here, linear SVM)
  clf = SGDClassifier(verbose=0, loss="hinge", alpha=hyper_param1, max_iter=1000, penalty="l2", random_state=0)

  # train
  clf.fit(train_X, train_Y)

  # validation
  valid_Y_hat = clf.predict(valid_X)

  accuracy = np.sum((valid_Y_hat == valid_Y)) / 200.0 * 100.0
  print('validation accuracy = ' + str(accuracy) + ' %')

  return clf, accuracy


def train_xgb(train_X, train_Y, valid_X, valid_Y, random_state):
  train_X = np.concatenate([train_X, valid_X])
  train_Y = np.concatenate([train_Y, valid_Y])

  best_params = {'colsample_bytree': 0.8208300279019551,
                 'gamma': 0.018782526251964193,
                 'learning_rate': 0.32538948689079206,
                 'max_depth': 2,
                 'n_estimators': 135,
                 'subsample': 0.975054090135046}

  clf = xgb.XGBClassifier(**best_params, random_state=random_state).fit(train_X, train_Y)

  """
  from scipy.stats import uniform, randint
  from sklearn.model_selection import RandomizedSearchCV
  
  params = {
    "colsample_bytree": uniform(0.7, 0.3),
    "gamma": uniform(0, 0.5),
    "learning_rate": uniform(0.03, 0.3),  # default 0.1
    "max_depth": randint(2, 6),  # default 3
    "n_estimators": randint(100, 150),  # default 100
    "subsample": uniform(0.6, 0.4)
  }

  search = RandomizedSearchCV(clf, param_distributions=params, random_state=0, n_iter=200, cv=5, verbose=1,
                              n_jobs=-1, return_train_score=True)

  search.fit(train_X, train_Y)

  clf = search.best_estimator_

  print(search.best_score_)
  print(search)
  print(search.best_params_)
  """

  return clf


if __name__ == '__main__':
  # load data
  train_X = mean_mfcc('train')
  valid_X = mean_mfcc('valid')
  test_X = mean_mfcc('test')

  # label generation
  cls = np.arange(0, 10)
  train_Y = np.repeat(cls, 100)
  valid_Y = np.repeat(cls, 20)
  test_Y = np.repeat(cls, 20)

  # feature normalizaiton
  train_X = train_X.T
  train_X_mean = np.mean(train_X, axis=0)
  train_X = train_X - train_X_mean
  train_X_std = np.std(train_X, axis=0)
  train_X = train_X / (train_X_std + 1e-5)

  valid_X = valid_X.T
  valid_X = valid_X - train_X_mean
  valid_X = valid_X / (train_X_std + 1e-5)

  # Use validation set for training
  train_X = np.concatenate([train_X, valid_X])
  train_Y = np.concatenate([train_Y, valid_Y])

  best_params = {'colsample_bytree': 0.8208300279019551,
                 'gamma': 0.018782526251964193,
                 'learning_rate': 0.32538948689079206,
                 'max_depth': 2,
                 'n_estimators': 135,
                 'subsample': 0.975054090135046}

  # clfs = [(str(i), xgb.XGBClassifier(**best_params, random_state=i)) for i in range(100)]
  # clf = VotingClassifier(estimators=clfs, n_jobs=-1).fit(train_X, train_Y)

  clf = xgb.XGBClassifier(**best_params, random_state=1).fit(train_X, train_Y)

  # now, evaluate the model with the test set
  test_X = test_X.T
  test_X = test_X - train_X_mean
  test_X = test_X / (train_X_std + 1e-5)
  test_Y_hat = clf.predict(test_X)

  accuracy = np.sum((test_Y_hat == test_Y)) / 200.0 * 100.0
  print('test accuracy = ' + str(accuracy) + ' %')
  print()

  fi = clf.feature_importances_
  fi_order = fi.argsort()[::-1]
  plt.figure(figsize=(15, 35))
  sns.barplot(x=fi[fi_order], y=FEATURE_NAMES[fi_order])
  plt.tight_layout()
  plt.savefig('feature_importance.pdf')
  plt.show()

  print()

  #
  #
  # # training model
  # alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
  #
  # model = []
  # valid_acc = []
  # for a in alphas:
  #   clf, acc = train_model(train_X, train_Y, valid_X, valid_Y, a)
  #   model.append(clf)
  #   valid_acc.append(acc)
  #
  # # choose the model that achieve the best validation accuracy
  # final_model = model[np.argmax(valid_acc)]
  #
  # # now, evaluate the model with the test set
  # test_X = test_X.T
  # test_X = test_X - train_X_mean
  # test_X = test_X / (train_X_std + 1e-5)
  # test_Y_hat = final_model.predict(test_X)
  #
  # accuracy = np.sum((test_Y_hat == test_Y)) / 200.0 * 100.0
  # print('test accuracy = ' + str(accuracy) + ' %')
