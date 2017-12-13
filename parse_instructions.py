"""
parse_instructions.py

Usage:
	python parse_instructions.py jsonDir

	jsonDir - a directory containing the scraped data from allrecipes in JSON format (get_recipe_information.py)

Function:
	Generates processed/instructions.json, which maps recipeID to counts of the imperatives in its instructions
"""

import os, sys
import json
import nltk
import collections
import re

# Returns map from recipeIDs -> imperatives -> counts
def main():

	# Get name of directory containing scraped JSON data
	jsonDir = sys.argv[1]

	# Initialize empty dict of dict: {recipe id -> {imperatives -> counts}}
	imperative_map = {}

	# Regular Expression for capturing and removing non-ascii characters
	reg = re.compile('[^\x00-\x7F]+')

	# Loop over each file containing recipe data
	for filename in os.listdir(jsonDir):
		if filename.endswith('.json'):

			# Get all dict of all recipes' JSON data from this file
			data = json.load(open(os.path.join(jsonDir, filename)))

			# Get all dict of all recipes' JSON data from this file
			for recipe_id, recipe in data.iteritems():

				# Get list of instructions as one string
				instructions_lg = ' '.join(recipe['instructions'])

				# Initialize empty dict: {imperatives -> counts} for this recipe
				imperatives = collections.Counter()

				# Sentence-tokenize the instructions
				sentences = nltk.sent_tokenize(instructions_lg)

				# Iterate through sentences
				for s in sentences:

					# Loop through each raw clause in current sentence
					for raw_clause in s.split('; '):
						if not raw_clause:
							continue

						# Remove non-ascii characters
						raw_clause = re.sub(reg,' ', raw_clause)
						
						# Append the word 'You' to each clause to encourage imperative verbs
						# at the beggining of sentences to be tagged as verbs, and not nouns
						clause = 'You {}'.format(raw_clause.rstrip('.'))

						# Word-tokenize the words in the current sentence
						words = nltk.word_tokenize(clause)
						
						# Discount sentence if too short
						if len(words) == 1:
							continue
						
						# Lower case words in the sentence
						words[1] = words[1].lower()

						# Tag the parts of speech of each word
						pos_labels = nltk.pos_tag(words)
						
						# Collect counts of all words tagged as imperative verbs 
						for label in pos_labels:
							if label[1] == 'VBP':
								imperatives[label[0].lower()] += 1

				# Store the dict {imperative -> count} as this value for key=recipe_id in imperative_map
				imperative_map[recipe_id] = imperatives

	# Write the full processed JSON data to its permanent location processed/instriuctions.json
	filename = 'processed/instructions.json'
	with open(filename, 'w') as fp:
		json.dump(imperative_map, fp, sort_keys=True, indent=4, separators=(',', ': '))

if __name__ == '__main__':
	main()