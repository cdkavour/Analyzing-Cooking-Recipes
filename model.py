from sklearn.tree import DecisionTreeRegressor

import sys
import numpy as np
from sklearn.utils import resample
import extract_features


def get_accuracy(y_test, y_pred):
    return np.average((np.absolute(np.subtract(y_test, y_pred))))

def train_random_forest(train_x, train_y, m, n_clf=10):
    """
    Returns accuracy on the test set test_x with corresponding labels test_y
    using a random forest classifier with n_clf decision trees trained with
    training examples train_x and training labels train_y

    Input:
        train_x : np.array (n_train, d) - array of training feature vectors
        train_y : np.array (n_train) = array of labels corresponding to train_x samples

        m : int - number of features to consider when splitting
        n_clf : int - number of decision tree classifiers in the random forest, default is 10

    Returns:
        accracy : float - accuracy of random forest classifier on test_x samples
    """

    n, d = train_x.shape

    forest = []

    for i in range(n_clf):
        #select features
        forest.append(DecisionTreeRegressor(max_features=m))
        x_train_sample, y_train_sample = resample(train_x, train_y, n_samples=n)
        forest[i].fit(x_train_sample, y_train_sample)

    return forest


def test_random_forest(test_x, y_true, forest):
    """
    Input:
        test_x : np.array (n_test, d) - array of testing feature vectors
        test_y : np.array (n_test) - array of labels corresponding to test_x samples
    """

    n_test, d_test = test_x.shape
    n_clf = len(forest)
    forest_pred = np.zeros((n_clf, n_test))

    for j in range(n_clf):
        pred = forest[j].predict(test_x)
        forest_pred[j] = pred

    y_pred = np.average(np.transpose(forest_pred), axis=1)

    return get_accuracy(y_true, y_pred)


def main():
    print("Getting training data")
    train_x, train_y = extract_features.get_data()

    print("Getting testing data")
    test_x, y_true = extract_features.get_data()

    print("Training Model")
    forest = train_random_forest(train_x, train_y, 2)

    print("Testing Model")
    accuracy = test_random_forest(test_x, y_true, forest)

    print("Average minutes incorrect: " + str(accuracy))

if __name__ == '__main__':
    main()
