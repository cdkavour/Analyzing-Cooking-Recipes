import numpy as np
#import ipdb
from sets import Set

def create_feature(ids, parse_tree):
	'''
		INPUT:
		imperatives - a dictionary of recipe ids to imperatives to counts
	'''
	unique_items = []
	for id in ids:
		for item in parse_tree[id].keys():
			if item not in unique_items:
				unique_items.append(item)

	#print(unique_items)

	fv = np.zeros((len(ids), len(unique_items)))
	for i, id in enumerate(ids):
		for item in parse_tree[id].keys():
			j = unique_items.index(item)
			fv[i][j] = parse_tree[id][item]

	return fv

def create_time_feature(ids, times):
	y = np.zeros(len(ids))
	for i, id in enumerate(ids):
		y[i] = times[id]
	return y

def generate_features(imperatives, ingredients, times, num_instructions, num_ingredients):
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
	x = np.concatenate((f1, f2, f3, f4), axis=1)
	y = create_time_feature(ids, times)

	return x, y