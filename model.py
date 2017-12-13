'''
model.py

Module Description: This module defines the machine learning model(s) used to predict recipe
					Ready In Time, and executes it/them on the recipe feature data parsed by the 
					parsing scripts from the scraped input recipe data.
'''

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
import math

############################# HELPER FUNCTIONS ################################

def get_accuracy(y_test, y_pred):
	'''
		INPUT:
			y_test: true labels of recipe times
			y_pred: predicted labels of recipe times

		OUTPUT:
			acc: accuracy of your predictions
	'''

	# Calculate vector of the time errors (absolute value of (predicted time - true time) ) for each prediction
	diff = np.absolute(np.subtract(y_test, y_pred))

	# Calculate vector of the percent errors for each prediction
	percentage = np.divide(diff, y_test)

	''' Plot Results '''
	# Create scatter plot of our predicted times against true times
	fig = plt.figure()
	plt.scatter(y_test, y_pred)

	# Include line indicating the median prep time across all labels
	plt.plot(np.full(y_test.shape, np.median(y_test)), color='red')

	# Include line indicating perfect preditions (y = x)
	plt.plot(np.arange(np.max(y_test)))

	# Show plot
	plt.show()

	# Save plot
	fig.savefig('accuracy.png')

	# Return the average percent error across all predictions
	acc = np.average(percentage) * 100
	return acc

def train_model(train_x, train_y, m, n_clf=1):
	"""
	Returns accuracy on the test set test_x with corresponding labels test_y
	using a model of our choice, trained with
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
	model = []

	for i in range(n_clf):
		#model.append(DecisionTreeRegressor(max_features=m))
		#model.append(KNeighborsRegressor(n_neighbors=m))
		#model.append(GaussianProcessRegressor(alpha=m))
		model.append(svm.SVR(C=m))
		x_train_sample, y_train_sample = resample(train_x, train_y, n_samples=n)
		model[i].fit(x_train_sample, y_train_sample)

	return model


def test_model(test_x, y_true, model):
	"""
	Input:
		test_x : np.array (n_test, d) - array of testing feature vectors
		test_y : np.array (n_test) - array of labels corresponding to test_x samples
	"""

	n_test, d_test = test_x.shape
	n_clf = len(model)

	model_pred = np.zeros((n_clf, n_test))

	for j in range(n_clf):
		pred = model[j].predict(test_x)
		model_pred[j] = pred

	y_pred = np.ceil(np.average(np.transpose(model_pred), axis=1))

	return get_accuracy(y_true, y_pred)

def baseline(truths):
	median = np.median(truths)
	acc = get_accuracy( truths, np.full(truths.shape, median))
	print('Baseline: {}'.format(acc))

############################# MAIN ################################

def main():
	print("Getting training data")

	# Get parsed json data as dictionary objects for inputs to the model
	imperatives = extra_functions.json_to_dict("processed/instructions.json")
	ingredients = extra_functions.json_to_dict("processed/ingredients.json")
	num_instructions = extra_functions.json_to_dict("processed/num_instructions.json")
	num_ingredients = extra_functions.json_to_dict("processed/num_ingredients.json")
	instruction_times = extra_functions.json_to_dict("processed/instruction_time.json")
	times = extra_functions.json_to_dict("processed/times.json")

	# Get ordered list of recipe ids
	recipeIDs = times.keys()

	# Exclude recipes who's true Ready-In Times are greater than 24 hours
	for recipeID in recipeIDs:
		if times[recipeID] > 24*60: del times[recipeID]

	# Get feature matrix x, and true label vector y
	x, y, ids = extract_features.generate_features(imperatives, ingredients, times, num_instructions, num_ingredients, instruction_times)
	s = np.arange(len(x))
	np.random.shuffle(s)
	x = x[s]
	y = y[s]
	ids = ids[s]

	# Split data into train, test data (70% train data)
	train_split = int(len(x))/10*7
	train_x, train_y = x[:train_split], y[:train_split]
	test_x, test_y = x[train_split:], y[train_split:]
	train_ids, test_ids = ids[train_split:], ids[train_split:]

	# Get the baseline ready-in time prediction (median across the training data)
	baseline(train_y)

	# Set hyper-parameters which we deemed best (results of test_hyperparameters.py script)
	c = 100 # regularization parameter
	m = 200 # number o
	f = 3 # number of individual models used for our overarching model, which averages over the results of each

	# Create 5-folds for cross validation
	k = int(math.ceil(train_split/5.0))
	train_x_folds = [train_x[1:k], train_x[k:2*k], train_x[2*k:3*k], train_x[3*k:4*k], train_x[4*k:]]
	train_y_folds = [train_y[1:k], train_y[k:2*k], train_y[2*k:3*k], train_y[3*k:4*k], train_y[4*k:]]

	# Run model on each fold
	best_forest = []
	best_accuracy = 0
	for fold in range(5):
		train_x_current = [train_x[0]]
		train_y_current = [train_y[0]]


		for i in range(5):
			if fold != i:
				train_x_current = np.concatenate((train_x_current, train_x_folds[i]))
				train_y_current = np.concatenate((train_y_current, train_y_folds[i]))


		model = train_model(train_x_current, train_y_current, c, f)
		accuracy = test_model(train_x_folds[fold], train_y_folds[fold], model)
		print("FOLD # " + str(fold+1) + " " + str(accuracy))

		if accuracy > best_accuracy:
			best_accuracy = accuracy
			best_model = forest

	final_acc = test_model(test_x, test_y, best_forest)
	print("TEST: " + str(final_acc))


if __name__ == '__main__':
	main()
