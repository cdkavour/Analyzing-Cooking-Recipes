from sklearn.tree import DecisionTreeRegressor

import sys
import numpy as np
from sklearn.utils import resample
import extract_features
#import extra_functions

def get_accuracy(y_test, y_pred):
    return np.average((np.square(np.subtract(y_test, y_pred))))

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
    #imperatives = extra_functions.json_to_dict("processed/instructions.json")
    #ingredients = extra_functions.json_to_dict("processed/ingredients.json")
    #times = extra_functions.json_to_dict("processed/times.json")

    imperatives = {
        "1": {
            "cook": 1,
            "bake": 1
        },
        "2": {
            "cook": 2,
            "clean": 1
        },
        "3": {
            "bake": 3,
            "cut": 2 
        }
    }
    ingredients = {
        "1" : {
            "pork": 1,
            "carrots": 3
        },
        "2" : {
            "carrots": 1,
            "tofu": 2
        },
        "3" : {
            "zucchini": 2,
            "henry": 3
        }
    }
    times = {
        "1": "20",
        "2": "15",
        "3": "10"
    }

    x, y = extract_features.generate_features(imperatives, ingredients, times)
    train_split = int(len(x))/3*2
    print(x)
    print(y)
    train_x, train_y = x[:train_split], y[:train_split]
    test_x, test_y = x[train_split:], y[train_split:]
    print("Training Model")
    print(train_x)
    print(train_y)
    print(test_x)
    print(test_y)
    forest = train_random_forest(train_x, train_y, 2)

    print("Testing Model")
    accuracy = test_random_forest(test_x, test_y, forest)

    print("Average minutes incorrect: " + str(accuracy))

if __name__ == '__main__':
    main()
