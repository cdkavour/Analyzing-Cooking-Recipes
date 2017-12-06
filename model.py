from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
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

	#Plotting
	fig = plt.figure()
	plt.scatter(y_test, y_pred)
	plt.plot(np.full(y_test.shape, np.median(y_test)), color='red')
	plt.plot(np.arange(np.max(y_test)))
	plt.show()
	fig.savefig('accuracy.png')

	return np.average(percentage) * 100

def train_random_forest(train_x, train_y, m, n_clf=1):
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
		#forest.append(DecisionTreeRegressor(max_features=m))
		forest.append(svm.SVR(C=m))
		#forest.append(KNeighborsRegressor(n_neighbors=m))
		#forest.append(GaussianProcessRegressor(alpha=m))
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

	y_pred = np.ceil(np.average(np.transpose(forest_pred), axis=1))

	return get_accuracy(y_true, y_pred)

def baseline(truths):
	median = np.median(truths)
	#print(median)

	acc = get_accuracy( truths, np.full(truths.shape, median))

	print('Baseline: {}'.format(acc))

def main():
	print("Getting training data")
	imperatives = extra_functions.json_to_dict("processed/instructions.json")
	ingredients = extra_functions.json_to_dict("processed/ingredients.json")
	times = extra_functions.json_to_dict("processed/times.json")
	num_instructions = extra_functions.json_to_dict("processed/num_instructions.json")
	num_ingredients = extra_functions.json_to_dict("processed/num_ingredients.json")
	instruction_times = extra_functions.json_to_dict("processed/instruction_time.json")

	recipeIDs = times.keys()
	for recipeID in recipeIDs:
		if times[recipeID] > 24*60: del times[recipeID]


	x, y, ids = extract_features.generate_features(imperatives, ingredients, times, num_instructions, num_ingredients, instruction_times)
	s = np.arange(len(x))
	np.random.shuffle(s)
	x = x[s]
	y = y[s]
	ids = ids[s]

	m = 200
	#f_range = (np.arange(10)+1)*5
	#f_range = np.power((np.arange(10)+1), math.e).astype(int)
	f = 3

	#c_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000] #C = 100 / 1000
	#n_n_range = [3, 6, 9, 12, 15, 18, 21]
	#alpha_range = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8]

	train_split = int(len(x))/10*7
	train_x, train_y = x[:train_split], y[:train_split]
	test_x, test_y = x[train_split:], y[train_split:]
	train_ids, test_ids = ids[train_split:], ids[train_split:]
	baseline(train_y)

	c = 100
	best_forest = []
	best_accuracy = 0

	k = int(math.ceil(train_split/5.0))
	train_x_folds = [train_x[1:k], train_x[k:2*k], train_x[2*k:3*k], train_x[3*k:4*k], train_x[4*k:]]
	train_y_folds = [train_y[1:k], train_y[k:2*k], train_y[2*k:3*k], train_y[3*k:4*k], train_y[4*k:]]

	for fold in range(5):
		train_x_current = [train_x[0]]
		train_y_current = [train_y[0]]

		for i in range(5):
			if fold != i:
				train_x_current = np.concatenate((train_x_current, train_x_folds[i]))
				train_y_current = np.concatenate((train_y_current, train_y_folds[i]))

		forest = train_random_forest(train_x_current, train_y_current, c, f)
		accuracy = test_random_forest(train_x_folds[fold], train_y_folds[fold], forest)
		print("FOLD # " + str(fold+1) + " " + str(accuracy))

		if accuracy > best_accuracy:
			best_accuracy = accuracy
			best_forest = forest

	final_acc = test_random_forest(test_x, test_y, best_forest)
	print("TEST: " + str(final_acc))

	#print("C = {0}: {1}".format(alpha, np.average(acc)))


if __name__ == '__main__':
	main()
