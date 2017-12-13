"""
parse_ingredients.py

Usage:
	python parse_ingredients.py jsonDir

	jsonDir - a directory containing the scraped data from allrecipes in JSON format (get_recipe_information.py)

Function:
	Generates processed/ingredients.json, which maps recipeID to counts of its ingredients
	** Relies on the NYTimes ingredient tagger being installed **
"""

import os, sys
import json
import collections
import re

def main():

	# Get name of directory containing scraped JSON data
	jsonDir = sys.argv[1]

	# Hard coded relative path for Ingredient tagger
	relPath = '../ingredient-phrase-tagger-master/ingredient-phrase-tagger-master/bin/'

	# Regular Expression for capturing and removing non-ascii characters
	reg = re.compile('[^\x00-\x7F]+')

	# Initialize empty dict of dict: {recipe id -> {ingredients -> counts}}
	id_map = {}

	# Initialize empty dict: {ingredients -> counts} over all recipes
	allIngCount = collections.Counter()

	# Loop over each file containing recipe data
	for filename in os.listdir(jsonDir):
		if filename.endswith('.json'):

			# Get all dict of all recipes' JSON data from this file
			data = json.load(open(os.path.join(jsonDir, filename)))

			# Loop through each recipe from the current file
			for recipe_id, recipe in data.iteritems():

				# Open file for writing intermediate data (will be removed when script is done)
				ingFile = open('tmp/ingredients.in', 'wb')

				# Loop through each ingredient in the current recipe
				for ingredient in recipe['ingredients']:
					if not ingredient:
						continue

					# Remove non-ascii characters
					ingredient = re.sub(reg,' ', ingredient)

					# Write ingredient to intermediate output file
					ingFile.write(ingredient + '\n')

				# Close intermediate output file
				ingFile.close()

				# Run the ingredient parser on intermediate output of ingredients, convert to json, and put
				# results in temporary folder tmp/results.json
				os.system('python {}parse-ingredients.py tmp/ingredients.in > tmp/ingredients.out'.format(relPath))
				os.system('python {}convert-to-json.py tmp/ingredients.out > tmp/results.json'.format(relPath))

				# Initialize empty dict: {ingredients -> counts} for this recipe
				ingCount = collections.Counter()

				# Open temporary results file and load results into a dict
				resultFile = open('tmp/results.json')
				parsed = json.load(resultFile)

				# Sum up the counts of ingredients from the temporary results
				for entry in parsed:
					if 'name' in entry:
						ingCount[entry['name'].lower()] += 1

				# Store the dict {ingredient -> count} as this value for key=recipe_id in id_map
				id_map[recipe_id] = ingCount

				# Add counts from {ingredient -> count} for this recipe to all recipe counts
				allIngCount += ingCount

	# Print the most common ingredients seen across all recipes
	print(allIngCount.most_common(20))

	# Write the full processed JSON data to its permanent location processed/ingredients.json
	filename = 'processed/ingredients.json'
	with open(filename, 'w') as fp:
		json.dump(id_map, fp, sort_keys=True, indent=4, separators=(',', ': '))

if __name__ == '__main__':
	main()