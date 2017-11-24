import pandas as pd
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
import redis
from tqdm import tqdm
import argparse


def loading_data(file_path):
    # avoid to reload EVER time the full json.
    r = redis.StrictRedis(host='localhost', port=6379, db=0)

    # check if it's in redis
    if r.get(file_path) is None:
        print("Start loading data...")
        # we load the data
        X, Y = loading_json(file_path)
        # and save it to redis
        r.set(file_path, True)
        r.set(file_path + 'X', X.to_msgpack(compress='zlib'))
        r.set(file_path + 'Y', Y.to_msgpack(compress='zlib'))
        print("End loading data.")
    else:
        X = pd.read_msgpack(r.get(file_path + 'X'))
        Y = pd.read_msgpack(r.get(file_path + 'Y'))

    return X, Y


def loading_json(file_path):
    # we load the json data into dict
    data = json.load(open(file_path))
    # convert this dict into dataframe. We also transpose the dataframe, so
    # every line is an item. We change the NaN value for a -1 number.
    df = pd.DataFrame(data).transpose().fillna(-1)
    # get all the columns that aren't object
    features = df.dtypes[df.dtypes != object].index
    # we know the target is tool
    target = 'Tool'

    X = df[features]
    Y = df[target]
    return X, Y


def bagging_knn(train_X, train_Y, test_X, test_Y):
    # we declare the bagging. It will contains n_estimators of
    # KNeighborsClassifier(n_neighbors=10, weights='distance', n_jobs=4).
    # Every knn-classifier will take the half of the training set to learn.
    bagging = BaggingClassifier(
        KNeighborsClassifier(
            n_neighbors=10, weights='distance', n_jobs=4
        ), max_samples=0.5, max_features=0.5, n_estimators=10
    )

    # train the model
    bagging.fit(train_X, train_Y)

    # return the accuracy of guessing
    return bagging.score(test_X, test_Y)


def parsing_args():
    parser = argparse.ArgumentParser(
        description='Find the best Tool for a given model.'
    )
    parser.add_argument(
        '--file_path', type=str, help='The path to file containing the data'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    # manage the args
    args = parsing_args()

    if args.file_path is None:
        raise ValueError("You need to feed a path file")

    file_path = args.file_path
    # we load the data
    X, Y = loading_data(file_path)

    # we map the string class to int class
    classes = np.unique(Y.iloc[:])
    classes = {v: k for k, v in dict(enumerate(classes.flatten(), 1)).items()}

    # we change the classes in Y too
    Y.replace(classes, inplace=True)

    mean_accuracy = []
    for _ in tqdm(range(20)):
        n = X.shape[0]
        # get train_size element for the training
        train_size = int(n * 0.75)
        # sampling the elements
        train_X = X.sample(train_size)
        train_Y = Y[train_X.index]

        # remove the train point
        tmp_X = pd.concat([X, train_X]).drop_duplicates(keep=False)
        # get the test points
        test_X = tmp_X.sample(min(int(n * 0.25), tmp_X.shape[0]))
        test_Y = Y[test_X.index]

        mean_accuracy.append(
            bagging_knn(train_X, train_Y, test_X, test_Y)
        )

    print(f"Mean accuracy = {sum(mean_accuracy) / len(mean_accuracy)}")
