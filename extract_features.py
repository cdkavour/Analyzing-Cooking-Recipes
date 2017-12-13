'''
extract_features.py

Module Description: Functions for extracting and returning features from input data

Helper functions:
	create_feature(ids, nested_dct),
	create_list_feature(ids, dct)

Main usage by model.py script:
	generate_features(imperatives, ingredients, times, num_instructions, num_ingredients, instruction_time)
'''

import numpy as np
from sets import Set

# Optional set of stopwords to be used creating features
stop_words = {}

############################# HELPER FUNCTIONS ################################

def create_feature(ids, nested_dct):
	'''
		REQUIRES:
			INPUT:
				ids: deterministically ordered list of recipe ids
				nested_dct: nested dictionary mapping recipe ids to keywords to counts of those keywords

			OUTPUT:
				fm: feature matrix of recipe ids by keywords; fm[i][j] gives the count of keyword j in recipe i
		MODIFIES:
			n/a
		EFFECTS:
			Creates a matrix of recipe/keyword counts fm over the recipes ids and their keyword values in nested_dct
	'''

	unique_items = []
	for ID in ids:
		for item in nested_dct[ID].keys():
			if item not in unique_items && not in stop_words:
				unique_items.append(item)

	fm = np.zeros((len(ids), len(unique_items)))
	for i, ID in enumerate(ids):
		for item in nested_dct[ID].keys():
			if item not in stop_words:
				j = unique_items.index(item)
				fm[i][j] = nested_dct[ID][item]

	return fm

def create_list_feature(ids, dct):
	'''
		REQUIRES:
			INPUT:
				ids: deterministically ordered list of recipe ids
				dct: a dictionary mapping of recipe ids to an integer feature or label

			OUTPUT:
				lst: vector output, indices of lst match the indicies of id values in ids,
					 lst[i] = dct value of the ith id in ids 
		MODIFIES:
			n/a
		EFFECTS:
			Creates a vector of recipes to integer valued featres from the dictionary object dct
	'''
	lst = np.zeros(len(ids))
	for i, ID in enumerate(ids):
		lst[i] = dct[ID]
	return lst

############################# GENERATE FEATURES ################################

def generate_features(imperatives, ingredients, times, num_instructions, num_ingredients, instruction_time):
	'''
		REQUIRES:
			INPUT:
				imperatives: a dictionary mapping of recipe ids to imperatives to counts
				ingredients: a dictionary mapping of recipe ids to ingredient objects
				times: a dictionary mapping of recipe ids to times
				num_instructions: a dictionary mapping from recipe ids to number of instructions
				num_ingredients: a dictionary mapping from recipe ids to number of ingredients

			OUTPUT:
				x: a numpy ndarray representing the feature matrix
					- Dimensions of x:
						# of rows - number of recipes
						# of columns - number of features
				y: a numpy ndarray representing the vector of true time labels of the recipe data
					- Dimensions of y:
						# of rows - number of recipes
				ids: deterministically ordered list of recipe ids
		MODIFIES:
			n/a
		EFFECTS:
			Returns a formatted feature matrix x, labels y, and an ordered list of ids from
			the parsed json input data, to inputs into our machine learning model(s) in model.py
	'''

	# Get ordered list of recipes
	ids = list(times.keys())

	# Create feature vectors f1,...,fn (each fi may have more than one column)
	f1 = create_feature(ids, imperatives)
	f2 = create_feature(ids, ingredients)
	f3 = create_list_feature(ids, num_instructions)
	f3 = f3.reshape(f3.shape[0], 1)
	f4 = create_list_feature(ids, num_ingredients)
	f4 = f4.reshape(f4.shape[0], 1)
	f5 = create_feature(ids, instruction_time)

	# Form full feature matrix from feature vectors above
	x = np.concatenate((f1, f2, f3, f4, f5), axis=1)

	# Implement smoothing on feature matrix
	x = np.multiply(np.divide(x, 100), 98)
	x_mask = np.multiply((x < 0.1).astype(int), .02)
	x = np.add(x, x_mask)

	# Get true time labels of each recipe
	y = create_list_feature(ids, times)

	return x, y, np.array(ids)
