"""
Usage:
	python parse_ingredients.py DIR

	DIR - a directory containing the scraped data from allrecipes in JSON format (get_recipe_information.py)

Function:
	Generates processed/times.json, which maps recipeID to ready times

"""
import os, sys
import json
import code #DEBUG code.interact(local=locals())

#Returns map from recipeIDs -> imperatives -> counts
def get_times(jsonDir):

	count_map = {}

	for filename in os.listdir(jsonDir):
		if filename.endswith('.json'):
			data = json.load(open(os.path.join(jsonDir, filename)))

			for recipe_id, recipe in data.iteritems():
				count_map[recipe_id] = recipe['ready']

	filename = 'processed/times.json'
	with open(filename, 'w') as fp:
		json.dump(count_map, fp, sort_keys=True, indent=4, separators=(',', ': '))


if __name__ == '__main__':
	inDir = sys.argv[1]
	get_times(inDir)