import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier


class MyAlgo(BaseEstimator, ClassifierMixin):
    """Custom classification algorithm. It can be choice when there is a big
    majority class. There is fit and score methods like in Scikit."""

    def __init__(self):
        self.binary = DecisionTreeClassifier()

        self.multi = DecisionTreeClassifier()

        self.majority_class = None
        self.classes = None

    def fit(self, training_x, training_y):
        """Training function. It takes a training vector features and a
        training class vector."""
        training_x = np.array(training_x)
        training_y = np.array(training_y)
        copy_y = training_y.copy()
        self.classes = np.unique(training_y)
        # we find the majority class
        self.majority_class = Counter(training_y).most_common()[0][0]
        # create a mask for the binary classification
        mask = copy_y == self.majority_class
        # apply the mask
        copy_y[mask] = self.majority_class
        copy_y[~mask] = 0
        self.binary.class_weight = {self.majority_class: 0.95, 0: 0.05}
        # fit the binary classifier if the mask is enough
        if np.any(mask):
            self.binary.fit(training_x, copy_y)
            # get the predictions
            y_pred = self.binary.predict(training_x)
            # filter the non majority class
            mask = y_pred != self.majority_class
            # fit on it
            self.multi.fit(training_x[mask], training_y[mask])
        else:
            self.multi.fit(training_x, training_y)

    def predict(self, test_x):
        """Predict function. It predict the class, based on given features
        vector."""
        test_x = np.array(test_x)
        y_pred = self.binary.predict(test_x)
        mask = y_pred != self.majority_class
        # to avoid the case of empty array
        if np.any(mask):
            y_pred[mask] = self.multi.predict(test_x[mask])
        return y_pred

    def score(self, X, y, sample_weight=None):
        """Score function. It computes the accuracy based on given features
        vector and class vector"""
        X = np.array(X)
        y = np.array(y)
        y_pred = self.predict(X)
        return np.sum(y_pred == y) / y.shape[0]

    def predict_proba(self, test_x):
        """Predict the probability of each classe to be the good one."""
        test_x = np.array(test_x)
        mask = self.classes == self.majority_class
        # multi probability
        y_pred = self.multi.predict_proba(test_x)
        # binary probability
        res_binary = self.binary.predict_proba(test_x)
        # multiply the binary and the multi on the right place
        y_pred[:, mask][:, 0] *= res_binary[:, 0]
        for index, line in enumerate(y_pred[:, ~mask]):
            y_pred[:, ~mask][index] = line * res_binary[:, 1][index]
        # return array of probability
        return y_pred

    def get_params(self, deep=True):
        """Return a dict with the parameters of the model"""
        # suppose this estimator has parameters "alpha" and "recursive"
        return {}

    def set_params(self, **parameters):
        """Set the parameters of the model"""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


def knn_distance(lhs, rhs, bound=2):
    """
    Fix the distance for knn, by taking into account that numbers greater
    than 1 represent enumerations.
    """
    # we suppose x and y have the same shape and are numpy array.
    # we create a mask for each vector. Values are true when it's
    # between [-1,+1]. The "*" operator is an "and"
    # We "and" the two mask and  change the values where the mask is True.
    lhs_mask = (lhs >= -1) * (lhs <= 1)
    rhs_mask = (rhs >= -1) * (rhs <= 1)
    diff = abs(lhs - rhs)
    diff[~(lhs_mask * rhs_mask)] = bound
    return diff.sum()


def init_algorithms(arguments):
    """
    Init all the machine learning algorithms. It returns a dictionnary with
    the name as key and an instance as value.
    """
    algorithms = {}

    # Classificator parts:
    # Do not include these algorithms with duplicates,
    # as they are very slow.
    if not arguments.duplicates:
        algorithms["knn"] = KNeighborsClassifier(
            n_neighbors=10,
            weights="distance",
            metric=knn_distance,
        )
        algorithms["bagging-knn"] = BaggingClassifier(
            KNeighborsClassifier(
                n_neighbors=10,
                weights="distance",
                metric=knn_distance
            ),
            max_samples=0.5,
            max_features=1,
            n_estimators=10,
        )

    algorithms["ada-boost"] = AdaBoostClassifier()

    algorithms["naive-bayes"] = GaussianNB()

    algorithms["svm"] = SVC()

    algorithms["linear-svm"] = LinearSVC()

    algorithms["decision-tree"] = DecisionTreeClassifier()

    algorithms["random-forest"] = RandomForestClassifier(
        n_estimators=30,
        max_features=None,
    )

    algorithms["neural-network"] = MLPClassifier(
        solver="lbfgs",
    )

    # Voting part:
    algorithms["voting-classifier"] = VotingClassifier(
        [
            ("decision-tree", DecisionTreeClassifier()),
            ("random-forest", RandomForestClassifier(
                n_estimators=30,
                max_features=None,
            )),
            ("svm", SVC(probability=True)),
        ],
        voting="soft"
    )

    # Custom part
    algorithms["custom-algo"] = MyAlgo()
    return algorithms
