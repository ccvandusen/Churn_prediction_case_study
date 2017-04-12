import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import roc_curve
import pandas as pd
import numpy as np
import data_engineer as de
import seaborn as sns
sns.set()


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


def cartesian(arrays, out=None):
    """
    Code borrowed from https://gist.github.com/glamp/5077283

    Generate a cartesian product of input arrays.
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in xrange(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out


def isolate_and_plot(X_train, y_train, variable):
    '''
    code in this function modified for use with this data from
    http://blog.yhat.com/posts/logistic-regression-and-python.html
    INPUT: pandas df, contains both dependent and independent variables
           str, variable name to plot on
    OUTPUT: matplotlib graph

    This function isolates the argument variable and the different cities in
    the dataset to see how prediction probabilites differ from city to city
    '''

    model = sm.Logit(np.asarray(y_train), X_train.astype(float)).fit()

    # Creating ranges of values to plot over
    avg_rating_of_driver = np.linspace(X_train['avg_rating_of_driver'].min(),
                                       X_train['avg_rating_of_driver'].max(), 10)
    luxury_car_user = np.linspace(X_train['luxury_car_user'].min(),
                                  X_train['luxury_car_user'].max(), 10)
    used_once = np.linspace(X_train['used_once'].min(),
                            X_train['used_once'].max(), 10)
    avg_distance = np.linspace(X_train['avg_dist'].min(),
                               X_train['avg_dist'].max(), 10)
    surge_pct = np.linspace(X_train['surge_pct'].min(),
                            X_train['surge_pct'].max(), 10)

    # Enumerating all combinations of variables
    combos = pd.DataFrame(cartesian([avg_rating_of_driver, luxury_car_user,
                                     used_once, avg_distance, surge_pct,
                                     [1, 2, 3],
                                     [1.]]))
    combos.columns = ['avg_rating_of_driver', 'luxury_car_user', 'used_once',
                      'avg_distance', 'surge_pct', 'cities', 'intercept']

    # Making dummy variables of cities
    dummy_ranks = pd.get_dummies(combos['cities'], prefix='city')
    dummy_ranks.columns = ['Astapor', "King's Landing", 'Andriod']
    cols_to_keep = ['avg_rating_of_driver', 'luxury_car_user', 'used_once',
                    'avg_distance', 'surge_pct', 'cities', 'intercept']
    combos = combos[cols_to_keep].join(dummy_ranks.ix[:, "King's Landing":])

    # Predicting on the enumerated dataset
    combos['churn_pred'] = model.predict(
        combos.ix[:, combos.columns != 'Astapor'])
    grouped = pd.pivot_table(combos, values=['churn_pred'], index=[variable,
                                                                   'cities'], aggfunc=np.mean)

    # Plotting the variable and cities against probabilites
    colors = 'rbgyrbgy'
    for col in combos.cities.unique():
        plt_data = grouped.ix[grouped.index.get_level_values(1) == col]
        plt.plot(plt_data.index.get_level_values(0), plt_data['churn_pred'],
                 color=colors[int(col)])
    plt.xlabel(variable)
    plt.ylabel("P(churn)")
    plt.legend(['Astapor', "King's Landing", 'Andriod'],
               loc='upper right', title='Cities')
    plt.title("P(churn) isolating " + variable + " and city")
    plt.savefig('avg_rating_of_driver.png')
    plt.show()

if __name__ == '__main__':
    train_filepath = '/Users/ChrisV/Documents/Galvanize/churn-prediction-case-study/data/churn_train.csv'
    test_filepath = '/Users/ChrisV/Documents/Galvanize/churn-prediction-case-study/data/churn_test.csv'
    train_df = de.import_data(train_filepath)
    test_df = de.import_data(test_filepath)
    X_train, y_train = de.feature_engineer(train_df)
    X_test, y_test = de.feature_engineer(test_df)
    isolate_and_plot(X_train, y_train, 'avg_rating_of_driver')
