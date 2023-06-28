#!/usr/bin/env python

import pandas as pd
import logging
import sys

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate


if __name__ == '__main__':
    logging.basicConfig(format="[%(levelname)s] %(asctime)s %(message)s",
            level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(f"Arguments count: {len(sys.argv)}")

    logger.info('reading training data')
    df_train = pd.read_csv('/mnt/beegfs/home/mpansare2020/credit-card-data/creditcard11.csv')
    logger.info(f'Total rows: {df_train.index}')

    features = [ "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount","Class"]

    X = df_train[features]
    y = df_train['Class']

    # drop Time from the feature list, as it is unique for the data row
    X = X.drop(['Time'], axis=1)

    classifier_names_1 = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
    ]

    classifiers_1 = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]

    classifier_names = [
        "Extra Trees",
        "Gradient Boosting",
        "Gradient Boosting-Regressor",
        "Histogram-Based Gradient-Boosting",
    ]

    classifiers = [
        ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0),
        GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),
        GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='squared_error'),
        HistGradientBoostingClassifier(max_iter=100),
    ]

    classifier_name = classifier_names[int(sys.argv[1])]
    clf = classifiers[int(sys.argv[1])]

    logger.info(f'fitting classifier - {classifier_name}')



    


  
