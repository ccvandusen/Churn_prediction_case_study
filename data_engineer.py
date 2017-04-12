import pandas as pd
import numpy as np

'''
This script has functions to clean the test and train data
as well as engineer new features, for use in the plots and churn_model
scripts
'''


def import_data(filepath):
    '''
    INPUT: STRING of filepath to data file
    OUTPUT: PANDAS DF with engineered churn indicators
    '''

    # Reading in data and making date columns as pandas datetime object
    full_df = pd.read_csv(filepath)
    full_df['last_trip_date'] = pd.to_datetime(full_df['last_trip_date'])
    full_df['signup_date'] = pd.to_datetime(full_df['signup_date'])

    # creating dependent churn variables
    # labelled customers churned if they hadn't used the service in the last
    # month
    condition = full_df['last_trip_date'] < '2014-06-01'
    full_df['churn'] = 1
    full_df.ix[~condition, 'churn'] = 0

    return full_df


def feature_engineer(full_df):
    '''
    INPUT: pandas dataframe - full data before split
    OUTPUT: Pandas Df - the X values of the data with new engineered columns
            Pandas Series - the y values to predict on
    '''

    # creating a feature of people who only used the service once
    used_once = []
    for num in xrange(len(full_df)):
        used_once.append(
            ((full_df['last_trip_date'][num] - full_df['signup_date'][num]).days > 2) * 1)
    full_df['used_once'] = pd.Series(used_once)

    # Filling missing values, see readme for our permuting method
    condition3 = full_df['avg_rating_of_driver'].isnull()
    full_df.ix[condition3, 'avg_rating_of_driver'] = 0.5 * \
        full_df['avg_rating_of_driver'].mean()

    condition4 = full_df['avg_rating_by_driver'].isnull()
    full_df.ix[condition4, 'avg_rating_by_driver'] = 0.5 * \
        full_df['avg_rating_by_driver'].mean()

    # Creating dependent variable object and dropping it from independent
    # variables
    y = full_df['churn']
    full_df.drop('churn', axis=1, inplace=True)

    # creating dummy variables
    full_df['city'].unique()
    df_city = pd.get_dummies(full_df['city'])
    full_df['phone'].unique()
    df_phone = pd.get_dummies(full_df['phone'])
    full_df = pd.concat([full_df, df_city], axis=1)
    full_df = pd.concat([full_df, df_phone], axis=1)

    # Dropping all extra columns not being used
    full_df.drop(full_df[['city', 'phone', 'avg_surge', 'iPhone', 'Winterfell',
                          'last_trip_date', 'signup_date',
                          'avg_rating_by_driver']],
                 axis=1, inplace=True)
    return full_df, y

'''
Some extra variables attempted but weren't used:
    full_df['no_ratings'] = \
        full_df['avg_rating_of_driver'].isnull()*1

    full_df['compatability'] = full_df['avg_rating_of_driver']\
        *full_df['avg_rating_by_driver']

    full_df['total_distance'] = full_df[\
        'trips_in_first_30_days'] * full_df['avg_dist']
'''
