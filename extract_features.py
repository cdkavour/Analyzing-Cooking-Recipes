import numpy as np
#import ipdb
from sets import Set
import code #DEBUG code.interact(local=locals())

stop_words = []

def create_feature(ids, parse_tree):
	'''
		INPUT:
		imperatives - a dictionary of recipe ids to imperatives to counts
	'''

	unique_items = []
	for ID in ids:
		for item in parse_tree[ID].keys():
			if item not in unique_items && not in stop_words:
				unique_items.append(item)
	#print(unique_items)

	fv = np.zeros((len(ids), len(unique_items)))
	for i, ID in enumerate(ids):
		for item in parse_tree[ID].keys():
			if item not in stop_words:
				j = unique_items.index(item)
				fv[i][j] = parse_tree[ID][item]

	return fv

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
	ids = list(times.keys())
	f1 = create_feature(ids, imperatives)
	f2 = create_feature(ids, ingredients)
	f3 = create_time_feature(ids, num_instructions)
	f3 = f3.reshape(f3.shape[0], 1)
	f4 = create_time_feature(ids, num_ingredients)
	f4 = f4.reshape(f4.shape[0], 1)
	f5 = create_feature(ids, instruction_time)
	x = np.concatenate((f1, f2, f3, f4, f5), axis=1)

	#smoothing
	x = np.multiply(np.divide(x, 100), 98)
	x_mask = np.multiply((x < 0.1).astype(int), .02)
	x = np.add(x, x_mask)

	y = create_time_feature(ids, times)

	return x, y, np.array(ids)
