"""
Usage:
	python parse_num_instructions.py DIR

	DIR - a directory containing the scraped data from allrecipes in JSON format (get_recipe_information.py)

Function:
	Generates processed/num_instructions.json, which maps recipeID to ready times
"""
import os, sys
import json
import code #DEBUG code.interact(local=locals())

#Returns map from recipeIDs -> imperatives -> counts
def get_num_instructions(jsonDir):

	count_map = {}

	for filename in os.listdir(jsonDir):
		if filename.endswith('.json'):
			data = json.load(open(os.path.join(jsonDir, filename)))

			for recipe_id, recipe in data.iteritems():
				count_map[recipe_id] = len(recipe['instructions'])

	filename = 'processed/num_instructions.json'
	with open(filename, 'w') as fp:
		json.dump(count_map, fp, sort_keys=True, indent=4, separators=(',', ': '))


if __name__ == '__main__':
	inDir = sys.argv[1]
	get_num_instructions(inDir)