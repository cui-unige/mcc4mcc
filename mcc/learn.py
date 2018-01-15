#! /usr/bin/env python3

import argparse
import json
import pandas
import statistics
import numpy as np
from tqdm                   import tqdm
from sklearn.neighbors      import KNeighborsClassifier
from sklearn.ensemble       import BaggingClassifier
from sklearn.naive_bayes    import GaussianNB
from sklearn.svm            import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier


def load(file_path):
    # Load the json data into dict:
    data = json.load(open(file_path))
    # Convert this dict into dataframe:
    df = pandas.DataFrame(data)
    # Select only the best tools:
    df = df.loc[df['Rank'] == 1]
    # remove the Tool columns from X.
    X = df.drop('Tool', 1)
    Y = df['Tool']
    return X, Y


def knn(train_X, train_Y, test_X, test_Y):
    clf = KNeighborsClassifier(
        n_neighbors=10, weights='distance', n_jobs=4
    )
    clf.fit(train_X, train_Y)

    return clf.score(test_X, test_Y)


def bagging_knn(train_X, train_Y, test_X, test_Y):
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


def gaussian_naive_bayes(train_X, train_Y, test_X, test_Y):
    # http://scikit-learn.org/stable/modules/naive_bayes.html
    gnb = GaussianNB()
    gnb.fit(train_X, train_Y)

    return gnb.score(test_X, test_Y)


def svm(train_X, train_Y, test_X, test_Y):
    # http://scikit-learn.org/stable/modules/svm.html
    # Support Vector Classifier with rbf kernel.
    clf = SVC()
    clf.fit(train_X, train_Y)

    return clf.score(test_X, test_Y)


def linear_svm(train_X, train_Y, test_X, test_Y):
    # http://scikit-learn.org/stable/modules/svm.html
    # Linear Support Vector classifier with rbf kernel.
    clf = LinearSVC()
    clf.fit(train_X, train_Y)

    return clf.score(test_X, test_Y)


def neural_net(train_X, train_Y, test_X, test_Y):
    # http://scikit-learn.org/stable/modules/neural_networks_supervised.html
    # Simple multi-layer neural net using Newton solver.
    nn = MLPClassifier(solver='lbfgs')
    nn.fit(train_X, train_Y)

    return nn.score(test_X, test_Y)


algorithms = {
    "KNN": knn,
    "Bagging+knn": bagging_knn,
    "Gaussian Naive Bayes": gaussian_naive_bayes,
    "Support Vector Machine": svm,
    "Linear Support Vector Machine": linear_svm,
    "Neural Net": neural_net,
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
