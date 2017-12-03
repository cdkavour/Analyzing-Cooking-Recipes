import os, sys
import json
import collections
import code #DEBUG code.interact(local=locals())

#Returns map from recipeIDs -> list of ingredients
def save_cook_nouns(jsonDir):

	relPath = '../ingredient-phrase-tagger-master/ingredient-phrase-tagger-master/bin/'

	for filename in os.listdir(jsonDir):
		if filename.endswith('.json'):
			data = json.load(open(os.path.join(jsonDir, filename)))

			for recipe_id, recipe in data.iteritems():

				ingFile = open('ingredients.in', 'w')
				for ingredient in recipe['ingredients']:
					ingFile.write(ingredient + '\n')
				ingFile.close()

				os.system('python {}parse-ingredients.py ingredients.in > ingredients.out'.format(relPath))
				os.system('python {}convert-to-json.py ingredients.out > results.json'.format(relPath))


if __name__ == '__main__':
	inDir = sys.argv[1]
	save_cook_nouns(inDir)