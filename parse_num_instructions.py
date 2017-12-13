"""
parse_num_instructions.py

Usage:
	python parse_num_instructions.py jsonDir

	jsonDir - a directory containing the scraped data from allrecipes in JSON format (get_recipe_information.py)

Function:
	Generates processed/num_instructions.json, which maps recipeID to ready times
"""
import os, sys
import json

# Returns map from recipeIDs -> imperatives -> counts
def main():

	# Get name of directory containing scraped JSON data
	jsonDir = sys.argv[1]

	# Initialize empty dict: {recipe id -> num_instructions}
	count_map = {}

	# Loop over each file containing recipe data
	for filename in os.listdir(jsonDir):
		if filename.endswith('.json'):

			# Get all dict of all recipes' JSON data from this file
			data = json.load(open(os.path.join(jsonDir, filename)))

			# Loop through each recipe from the current file
			for recipe_id, recipe in data.iteritems():

				# Get num instructions for this recipe
				count_map[recipe_id] = len(recipe['instructions'])

	# Write the full processed JSON data to its permanent location processed/num_instructions.json
	filename = 'processed/num_instructions.json'
	with open(filename, 'w') as fp:
		json.dump(count_map, fp, sort_keys=True, indent=4, separators=(',', ': '))

if __name__ == '__main__':
	main()