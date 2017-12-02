import os, sys
import json
import nltk
import collections

#Returns map from recipeIDs -> imperatives -> counts
def get_imperatives(jsonDir):

	imperative_map = {}

	for filename in os.listdir(jsonDir):
		if filename.endswith('.json'):
			data = json.load(open(os.path.join(jsonDir, filename)))

			for recipe_id, recipe in data.iteritems():
				instructions_lg = ' '.join(recipe['instructions'])

				sentences = nltk.sent_tokenize(instructions_lg)

				imperatives = collections.Counter()

				#Iterate through sentences
				for s in sentences:
					for raw_clause in s.split('; '):

						clause = 'They {}'.format(raw_clause.rstrip('.'))
						words = nltk.word_tokenize(clause)
						words[1] = words[1].lower()
						pos_labels = nltk.pos_tag(words)
						
						for label in pos_labels:
							if label[1] == 'VBP':
								imperatives[label[0]] += 1

						#TODO: Strictly pick verbs related to 'They' as the subject

				imperative_map[recipe_id] = imperatives

	print(imperative_map)

if __name__ == '__main__':
	inDir = sys.argv[1]
	get_imperatives(inDir)
