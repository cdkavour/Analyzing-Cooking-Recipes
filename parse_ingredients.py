#RUN IN CAEN LINUX

import os, sys
import json
import collections
import re
import code #DEBUG code.interact(local=locals())

#Returns map from recipeIDs -> list of ingredients
def save_cook_nouns(jsonDir):

	relPath = '../ingredient-phrase-tagger-master/ingredient-phrase-tagger-master/bin/'

	reg = re.compile('[^\x00-\x7F]+')

	id_map = {}

	for filename in os.listdir(jsonDir):
		if filename.endswith('.json'):
			data = json.load(open(os.path.join(jsonDir, filename)))

			for recipe_id, recipe in data.iteritems():

				ingFile = open('tmp/ingredients.in', 'wb')
				for ingredient in recipe['ingredients']:
					ingredient = re.sub(reg,' ', ingredient)
					ingFile.write(ingredient + '\n')
				ingFile.close()

				os.system('python {}parse-ingredients.py tmp/ingredients.in > tmp/ingredients.out'.format(relPath))
				os.system('python {}convert-to-json.py tmp/ingredients.out > tmp/results.json'.format(relPath))

				resultFile = open('tmp/results.json')
				parsed = json.load(resultFile)

				ingCount = collections.Counter()

				for entry in parsed:
					if 'name' in entry:
						ingCount[entry['name'].lower()] += 1

				id_map[recipe_id] = ingCount


	filename = 'processed/ingredients.json'
	with open(filename, 'w') as fp:
		json.dump(id_map, fp, sort_keys=True, indent=4, separators=(',', ': '))

if __name__ == '__main__':
	inDir = sys.argv[1]
	save_cook_nouns(inDir)