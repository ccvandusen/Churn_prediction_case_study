import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, \
    GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import roc_curve, precision_recall_fscore_support,\
    precision_score, recall_score
import seaborn as sns
import data_engineer as de

'''

This script has several functions that try different models on our
train and test data, as well as functions to plot ROC curve, feature importance,
and run a grid search on any of the fitted model objects. Data is cleaned/
engineered by functions in the data_engineer script.

'''

# So seaborn makes matplotlib plots pretty
sns.set()


def log_reg(X_train, y_train, X_test):
    '''
    INPUT: pandas objects of X_train, y_train, X_test data created from
           fxns in data_engineer script
    OUTPUT: fitted model object and probabilities created by model
    '''
    model = LogisticRegression().fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)
    return model, probabilities


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


def plot_ROC(probabilities, labels):
    '''
    INPUT: probabilities from any given model, numpy array of y_test data from
    test_train_split
    OUTPUT: Plotted ROC curve
    '''

    # Getting tpr and fpr to plot ROC curve from sk_learn
    fpr, tpr, thresholds = roc_curve(labels, probabilities[:, 1])

    # Plotting ROC curve
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity, Recall)")
    plt.title("Rideshare Gradient Boost ROC plot")
    plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), 'k-', zorder=0)
    plt.show()


def plot_feature_importance(model, X):
    '''
    INPUT: fitted model object, pandas df of indicator variables
    OUTPUT: Graph of feature importances for model
    '''
    importances = model.feature_importances_
    std = np.std([model.feature_importances_ for tree in model.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" %
              (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), list(X), rotation='vertical')
    plt.xlim([-1, X.shape[1]])
    plt.show()


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
    X_train, y_train = feature_engineer(train_df)
    X_test, y_test = feature_engineer(test_df)
    model, probabilities = gradient_boosting(X_train, y_train, X_test)
    plot_ROC(probabilities, y_test.values)
    #plot_feature_importance(model, X)
