import datetime
import pandas as pd
import numpy as np
import random
import time
import copy

random.seed(0)


def intersect(a, b):
    return list(set(a) & set(b))


def get_features(train, test):
    trainval = list(train.columns.values)
    testval = list(test.columns.values)
    output = intersect(trainval, testval)
    output.remove('people_id')
    output.remove('activity_id')
    return sorted(output)


def read_test_train():

    print("Read people.csv...")
    people = pd.read_csv("../all/people.csv",
                       dtype={'people_id': np.str,
                              'activity_id': np.str,
                              'char_38': np.int32},
                       parse_dates=['date'])

    print("Load train.csv...")
    train = pd.read_csv("../all/act_train.csv",
                        dtype={'people_id': np.str,
                               'activity_id': np.str,
                               'outcome': np.int8},
                        parse_dates=['date'])

    print("Load test.csv...")
    test = pd.read_csv("../all/act_test.csv",
                       dtype={'people_id': np.str,
                              'activity_id': np.str},
                       parse_dates=['date'])

    print("Process tables...")
    for table in [train, test]:
        table['year'] = table['date'].dt.year
        table['month'] = table['date'].dt.month
        table['day'] = table['date'].dt.day
        table.drop('date', axis=1, inplace=True)
        table['activity_category'] = table['activity_category'].str.lstrip('type ').astype(np.int32)
        for i in range(1, 11):
            table['char_' + str(i)].fillna('type -999', inplace=True)
            table['char_' + str(i)] = table['char_' + str(i)].str.lstrip('type ').astype(np.int32)

    people['year'] = people['date'].dt.year
    people['month'] = people['date'].dt.month
    people['day'] = people['date'].dt.day
    people.drop('date', axis=1, inplace=True)
    people['group_1'] = people['group_1'].str.lstrip('group ').astype(np.int32)
    for i in range(1, 10):
        people['char_' + str(i)] = people['char_' + str(i)].str.lstrip('type ').astype(np.int32)
    for i in range(10, 38):
        people['char_' + str(i)] = people['char_' + str(i)].astype(np.int32)

    print("Merge...")
    train = pd.merge(train, people, how='left', on='people_id', left_index=True)
    train.fillna(-999, inplace=True)
    test = pd.merge(test, people, how='left', on='people_id', left_index=True)
    test.fillna(-999, inplace=True)

    features = get_features(train, test)
    return train, test, features