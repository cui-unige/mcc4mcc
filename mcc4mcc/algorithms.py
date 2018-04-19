
"""
Machine learning algorithms used by mcc4mcc

This module returns a dictionary from identifiers to the algorithms,
as objects that conform to scikit-learn.
"""


from collections import Counter
import numpy
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier


class BMDT(BaseEstimator, ClassifierMixin):
    """
    Custom classification algorithm
    """

    def __init__(self):
        self.binary = DecisionTreeClassifier()
        self.multi = DecisionTreeClassifier()
        self.majority_class = None
        self.classes = None

    def fit(self, training_x, training_y):
        """
        Trains using the training vector features and a training class vector.
        """
        training_x = numpy.array(training_x)
        training_y = numpy.array(training_y)
        copy_y = training_y.copy()
        self.classes = numpy.unique(training_y)
        # Find the majority class:
        self.majority_class = Counter(training_y).most_common()[0][0]
        # Create a mask for the binary classification:
        mask = copy_y == self.majority_class
        # Apply the mask:
        copy_y[mask] = self.majority_class
        copy_y[~mask] = 0
        self.binary.class_weight = {self.majority_class: 0.95, 0: 0.05}
        # Fit the binary classifier if the mask is enough:
        if numpy.any(mask):
            self.binary.fit(training_x, copy_y)
            # Get the predictions:
            y_pred = self.binary.predict(training_x)
            # Filter the non majority class:
            mask = y_pred != self.majority_class
            # Fit on it:
            self.multi.fit(training_x[mask], training_y[mask])
        else:
            self.multi.fit(training_x, training_y)

    def predict(self, test_x):
        """
        Predicts the class, based on given features vector.
        """
        test_x = numpy.array(test_x)
        y_pred = self.binary.predict(test_x)
        mask = y_pred != self.majority_class
        # to avoid the case of empty array
        if numpy.any(mask):
            y_pred[mask] = self.multi.predict(test_x[mask])
        return y_pred

    def score(self, X, y, sample_weight=None):
        """
        Score computation.
        """
        X = numpy.array(X)
        y = numpy.array(y)
        y_pred = self.predict(X)
        return numpy.sum(y_pred == y) / y.shape[0]

    def predict_proba(self, test_x):
        """
        Predicts the probability of each classe to be the good one.
        """
        test_x = numpy.array(test_x)
        mask = self.classes == self.majority_class
        # Multi probability:
        y_pred = self.multi.predict_proba(test_x)
        # Binary probability:
        res_binary = self.binary.predict_proba(test_x)
        # Multiply the binary and the multi on the right place:
        y_pred[:, mask][:, 0] *= res_binary[:, 0]
        for index, line in enumerate(y_pred[:, ~mask]):
            y_pred[:, ~mask][index] = line * res_binary[:, 1][index]
        # Return array of probability
        return y_pred

    def get_params(self, deep=True):
        """
        Parameters of the model.
        """
        # Suppose this estimator has parameters "alpha" and "recursive"
        return {}

    def set_params(self, **parameters):
        """
        Sets the parameters of the model.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


def knn_distance(lhs, rhs, bound=2):
    """
    Fix the distance for knn, by taking into account that:
    * -1, 1 represent Boolean values (False, True),
    * 0 represents None,
    * values outside represent enumerations.
    """
    # We suppose x and y have the same shape and are numpy array.
    # We create a mask for each vector.
    # Values are true when it's between [-1,+1].
    # The "*" operator is an "and".
    # We "and" the two mask and change the values where the mask is True.
    lhs_mask = (lhs >= -1) * (lhs <= 1)
    rhs_mask = (rhs >= -1) * (rhs <= 1)
    diff = abs(lhs - rhs)
    diff[~(lhs_mask * rhs_mask)] = bound
    return diff.sum()


# Dictionary of algorithms to use.
#
# The "complex" key is a Boolean (or None) that tells if the algorithm
# should be used on big data sets.
ALGORITHMS = {}

ALGORITHMS["knn"] = KNeighborsClassifier(
    n_neighbors=10,
    weights="distance",
    metric=knn_distance,
)

ALGORITHMS["bagging-knn"] = BaggingClassifier(
    KNeighborsClassifier(
        n_neighbors=10,
        weights="distance",
        metric=knn_distance
    ),
    max_samples=0.5,
    max_features=1,
    n_estimators=10,
)

ALGORITHMS["ada-boost"] = AdaBoostClassifier()

ALGORITHMS["naive-bayes"] = GaussianNB()

ALGORITHMS["svm"] = SVC()

ALGORITHMS["linear-svm"] = LinearSVC()

ALGORITHMS["decision-tree"] = DecisionTreeClassifier()

ALGORITHMS["random-forest"] = RandomForestClassifier(
    n_estimators=30,
    max_features=None,
)

ALGORITHMS["neural-network"] = MLPClassifier(
    solver="lbfgs",
)

ALGORITHMS["voting-classifier"] = VotingClassifier(
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

ALGORITHMS["bmdt"] = BMDT()
