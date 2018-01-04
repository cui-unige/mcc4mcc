#! /usr/bin/env python3

import argparse
import json
import numpy
import pandas
import redis
import statistics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble  import BaggingClassifier
from tqdm              import tqdm
from hashlib           import md5

def load(file_path):
    digest = md5(open(file_path, 'rb').read()).hexdigest()
    client = redis.StrictRedis(host='localhost', port=6379, db=0)
    if client.get(digest) is None:
        # Load the json data into dict:
        data = json.load(open(file_path))
        # Convert this dict into dataframe:
        df = pandas.DataFrame(data)
        # Select only the best tools:
        df = df.loc[df['Rank'] == 1]
        X = df
        Y = df['Tool']
        # Save to redis:
        client.set(digest, True)
        client.set(digest + ':X', X.to_msgpack(compress='zlib'))
        client.set(digest + ':Y', Y.to_msgpack(compress='zlib'))
    else:
        # Read from redis:
        X = pandas.read_msgpack(client.get(digest + ':X'))
        Y = pandas.read_msgpack(client.get(digest + ':Y'))
    return X, Y

def bagging_knn (train_X, train_Y, test_X, test_Y):
    # Bagging contains n_estimators of
    # KNeighborsClassifier(n_neighbors=10, weights='distance', n_jobs=4).
    # Every knn-classifier will take the half of the training set to learn.
    bagging = BaggingClassifier(
        KNeighborsClassifier(
            n_neighbors=10, weights='distance', n_jobs=4
        ), max_samples=0.5, max_features=0.5, n_estimators=10
    )
    # Train the model:
    bagging.fit(train_X, train_Y)
    # Return the accuracy of guessing:
    return bagging.score(test_X, test_Y)

algorithms = {
    "Bagging+knn": bagging_knn,
}

if __name__ == '__main__':
    # Define command line:
    parser = argparse.ArgumentParser(
        description='Find the best Tool for a given model'
    )
    parser.add_argument(
        '--data',
        type=str,
        help='Path to data file',
        default='learning.json'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        help='Number of iterations',
        default=20,
    )
    args = parser.parse_args()
    # Load the data:
    X, Y = load(args.data)
    # Compute efficiency for each algorithm:
    results = dict()
    for algorithm in algorithms:
        subresults = []
        for _ in tqdm(range(args.iterations)):
            n             = X.shape[0]
            training_size = int(n * 0.75)
            training_X    = X.sample(training_size)
            training_Y    = Y[training_X.index]
            # Remove the training points:
            tmp_X         = pandas.concat([X, training_X]).drop_duplicates(keep=False)
            # Get the test points:
            test_X        = tmp_X.sample(min(int(n * 0.25), tmp_X.shape[0]))
            test_Y        = Y[test_X.index]
            subresults.append(algorithms.get(algorithm)(training_X, training_Y, test_X, test_Y))
        results[algorithm] = subresults
        print(f"Algorithm: {algorithm}")
        print(f"  Min     : {min                (subresults)}")
        print(f"  Max     : {max                (subresults)}")
        print(f"  Mean    : {statistics.mean    (subresults)}")
        print(f"  Median  : {statistics.median  (subresults)}")
        print(f"  Stdev   : {statistics.stdev   (subresults)}")
        print(f"  Variance: {statistics.variance(subresults)}")
