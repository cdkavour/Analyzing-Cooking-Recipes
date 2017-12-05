from sklearn.tree import DecisionTreeRegressor

import sys
import numpy as np
from sklearn.utils import resample
import extract_features
import extra_functions
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import code #DEBUG code.interact(local=locals())
#import ipdb
import math

def get_accuracy(y_test, y_pred):
	
	diff = np.absolute(np.subtract(y_test, y_pred))
	percentage = np.divide(diff, y_test)

	# #Plotting
	# fig = plt.figure()
	# plt.scatter(y_test, y_pred)
	# plt.plot(np.arange(700))
	# plt.show()


	# fig.savefig('accuracy.png')

	return np.average(percentage) * 100

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
	imperatives = extra_functions.json_to_dict("processed/instructions.json")
	ingredients = extra_functions.json_to_dict("processed/ingredients.json")
	times = extra_functions.json_to_dict("processed/times.json")
	num_instructions = extra_functions.json_to_dict("processed/num_instructions.json")
	num_ingredients = extra_functions.json_to_dict("processed/num_ingredients.json")

	x, y = extract_features.generate_features(imperatives, ingredients, times, num_instructions, num_ingredients)
	s = np.arange(len(x))
	x = x[s]
	y = y[s]

	m = 200
	#f_range = (np.arange(10)+1)*5
	#f_range = np.power((np.arange(10)+1), math.e).astype(int)
	f_range = [10]

	train_split = int(len(x))/10*7
	train_x, train_y = x[:train_split], y[:train_split]
	test_x, test_y = x[train_split:], y[train_split:]

	for f in f_range:
		acc = []
		#for i in range(40):
		forest = train_random_forest(train_x, train_y, m, f)
		accuracy = test_random_forest(test_x, test_y, forest)
		acc.append(accuracy)

		print("M = {0}: {1}".format(m, np.average(acc)))


if __name__ == '__main__':
	main()
