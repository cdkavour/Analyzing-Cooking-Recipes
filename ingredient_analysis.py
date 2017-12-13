from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
import sys
import numpy as np
from sklearn.utils import resample
import extra_functions
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

def create_feature(ids, parse_tree):
	'''
		INPUT:
		imperatives - a dictionary of recipe ids to imperatives to counts
	'''

	unique_items = []
	for ID in ids:
		for item in parse_tree[ID].keys():
			if item not in unique_items:
				unique_items.append(item)
	#print(unique_items)

	fv = np.zeros((len(ids), len(unique_items)))
	for i, ID in enumerate(ids):
		for item in parse_tree[ID].keys():
			j = unique_items.index(item)
			fv[i][j] = parse_tree[ID][item]

	return fv, unique_items

def create_time_feature(ids, times):
	y = np.zeros(len(ids))
	for i, ID in enumerate(ids):
		y[i] = times[ID]
	return y

def generate_features(imperatives, ingredients, times, num_instructions, num_ingredients, instruction_time):
	'''
		INPUT:
			imperatives - a dictionary of recipe ids to imperatives to counts
			ingredients - a dictionary of recipe ids to ingredient objects
			times - a dictionary of recipe ids to times
	'''
	features = []
	ids = list(times.keys())
	f1, fx = create_feature(ids, imperatives)
	features.extend(fx)
	f2, fx = create_feature(ids, ingredients)
	features.extend(fx)
	f3 = create_time_feature(ids, num_instructions)
	f3 = f3.reshape(f3.shape[0], 1)
	features.extend(["num_instructions"])
	f4 = create_time_feature(ids, num_ingredients)
	f4 = f4.reshape(f4.shape[0], 1)
	features.extend(["num_ingredients"])
	f5, fx = create_feature(ids, instruction_time)
	features.extend(fx)
	x = np.concatenate((f1, f2, f3, f4, f5), axis=1)
	n, d = x.shape

	#smoothing
	x = np.multiply(np.divide(x, 100), 98)
	x_mask = np.multiply((x < 0.1).astype(int), .02)
	x = np.add(x, x_mask)

	y = create_time_feature(ids, times)

	return x, y, np.array(ids), np.array(features)

def train_random_forest(train_x, train_y, c, n_clf=1):
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
		forest.append(svm.SVR(C=c))
		#forest.append(KNeighborsRegressor(n_neighbors=m))
		#forest.append(GaussianProcessRegressor(alpha=m))
		x_train_sample, y_train_sample = resample(train_x, train_y, n_samples=n)
		forest[i].fit(x_train_sample, y_train_sample)

	return forest


def test_random_forest(test_x, forest):
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

	return y_pred

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

	x, y, ids, features = generate_features(imperatives, ingredients, times, num_instructions, num_ingredients, instruction_times)
	n, d = x.shape
	s = np.arange(len(x))
	np.random.shuffle(s)
	x = x[s]
	y = y[s]
	ids = ids[s]

	train_x = x
	train_y = y
	test_x = np.zeros((d, d))
	np.fill_diagonal(test_x, 1)
	baseline_x = np.zeros((1, d))
	baseline_feature = ["baseline"]
	test_x = np.concatenate((test_x, baseline_x), axis=0)
	features = np.concatenate((features, baseline_feature), axis=0)

	c = 100
	f = 3
	runs = 1
	weights = np.zeros(d+1)
	for i in range(runs):
		forest = train_random_forest(train_x, train_y, c, f)
		y_pred = test_random_forest(test_x, forest)
		weights = np.add(weights, y_pred)

	weights = np.divide(weights, runs)
	p = weights.argsort()
	weights_sorted = weights[p]
	features_sorted = features[p]

	w = open('ingredient_analysis.txt', 'w')

	w.write("PREDICTING LOW VALUES\n")
	for i in range(20):
		w.write(features_sorted[i] + " " + str(weights_sorted[i]) + "\n")
	w.write('\n')
	w.write("PREDICTING HIGH VALUES\n")
	for i in range(20):
		w.write(features_sorted[len(weights)-1-i] + " " + str(weights_sorted[len(weights)-1-i]) + "\n")
	w.write('\n')

	weights_norm = np.absolute(np.subtract(weights, weights[d]))
	p_norm = weights_norm.argsort()
	weights_norm_sorted = weights_norm[p_norm]
	features_norm_sorted = features[p_norm]

	w.write('\n')
	w.write("MOST INFLUENTIAL VALUES\n")
	for i in range(40):
		w.write(features_norm_sorted[len(weights)-1-i] + " " + str(weights_norm_sorted[len(weights)-1-i]) + "\n")
	w.write('\n')


if __name__ == '__main__':
	main()
