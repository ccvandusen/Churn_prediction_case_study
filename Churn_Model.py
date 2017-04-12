import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, \
    GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import roc_curve, precision_recall_fscore_support,\
    precision_score, recall_score
import data_engineer as de

'''

This script has several functions that try different models on our
train and test data, as well as functions to plot ROC curve, feature importance,
and run a grid search on any of the fitted model objects. Data is cleaned/
engineered by functions in the data_engineer script.

'''


def log_reg(X_train, y_train, X_test):
    '''
    INPUT: pandas objects of X_train, y_train, X_test data created from
           fxns in data_engineer script
    OUTPUT: fitted model object
    '''
    model = LogisticRegression().fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)
    return model, probabilites


def random_forest(X_train, y_train, X_test):
    '''
    INPUT: pandas objects of X_train, y_train, X_test data created from
           fxns in data_engineer script
    OUTPUT: fitted model object and probabilities created by model
    '''
    model = RandomForestClassifier(n_estimators=50).fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)
    return model, probabilities


def gradient_boosting(X_train, y_train, X_test):
    '''
    INPUT: pandas objects of X_train, y_train, X_test data created from
           fxns in data_engineer script
    OUTPUT: fitted model object and probabilities created by model
    '''

    # this model has specific parameters, unlike the other functions, because
    # it was our final choice for prediction and the parameters were selected
    # from running the grid search fxn below
    model = GradientBoostingClassifier(n_estimators=200, max_features=1.0,
                                       learning_rate=0.1, max_depth=4,
                                       min_samples_leaf=17).fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)  # predict_proba() give 2
    # complementary probabilities

    return model, probabilities


def cross_val(model, X, y, folds=5, n_jobs=-1):
    '''
    INPUT: sk_learn object - fitted model to cross validate over
           panads df - train data of independent variables
           pandas df - train data of dependent variables
           int - number of folds, default 5
           int - number of jobs (cores/threads) to use, default -1 (all)
    OUTPUT: accuracy, precision, and recall validation scores
    '''

    accuracy = cross_val_score(
        model, X, y, cv=folds, n_jobs=n_jobs)
    precision = cross_val_score(
        model, X, y, cv=folds, n_jobs=n_jobs, scoring='precision')
    recall = cross_val_score(
        model, X, y, cv=folds, n_jobs=n_jobs, scoring='recall')
    print 'accuracy_scores : {}'.format(sum(accuracy) / len(accuracy))
    print 'precision_scores : {}'.format(sum(precision) / len(precision))
    print 'recall_scores : {}'.format(sum(recall) / len(recall))


def grid_search(model):
    '''
    INPUT: sklearn object: fitted model to grid Search
    OUTPUT: dict: contains the optimal parameters of the GridSearch
            float: accuracy score of the model w/ optimal parameters
    Implements a grid search
    '''
    # Parameters used to gridsearch, these are the optimized one for our final
    # Boosted model
    param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.05], 'max_depth': [
        1, 2, 4], 'min_samples_leaf': [9, 17], 'max_features': [1.0, 0.3]}

    # Gridsearch object
    gsearch1 = GridSearchCV(model, param_grid)

    return gsearch1.best_params_, gsearch1.best_score_

if __name__ == '__main__':
    train_filepath = '/Users/ChrisV/Documents/Galvanize/churn-prediction-case-study/data/churn_train.csv'
    test_filepath = '/Users/ChrisV/Documents/Galvanize/churn-prediction-case-study/data/churn_test.csv'
    train_df = de.import_data(train_filepath)
    test_df = de.import_data(test_filepath)
    X_train, y_train = de.feature_engineer(train_df)
    X_test, y_test = de.feature_engineer(test_df)
    model, probabilities = gradient_boosting(X_train, y_train, X_test)
    cross_val(model, X_train, y_train)
