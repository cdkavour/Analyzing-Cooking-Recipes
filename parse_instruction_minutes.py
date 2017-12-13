"""
parse_instruction_minutes.py

Usage:
	python parse_instruction_minutes.py jsonDir

	jsonDir - a directory containing the scraped data from allrecipes in JSON format (output from get_recipe_information.py)

Function:
	Generates processed/instructions.json, which maps recipeID to counts of the imperatives in its instructions
"""

import os, sys
import json
import nltk
import collections
from sets import Set
import re

# Returns map from recipeIDs -> imperatives -> counts
def main():

	# Get name of directory containing scraped JSON data
	jsonDir = sys.argv[1]

	# List of words representing time phrases in an instruction
	PL_TIME_WORDS = ['minutes', 'hours']
	WEIGHTS = [1, 1]

	# Initialize empty dict: {recipe id -> {time_word -> count}}
	time_map = {}

	# Loop over each file containing recipe data
	for filename in os.listdir(jsonDir):
		if filename.endswith('.json'):

			# Get all dict of all recipes' JSON data from this file
			data = json.load(open(os.path.join(jsonDir, filename)))

			# Loop through each recipe from the current file
			for recipe_id, recipe in data.iteritems():

				# Get list of instructions as one string
				instructions_lg = ' '.join(recipe['instructions'])

				# Initialize empty dict: {recipe id -> {time_word -> count}} for this recipe
				times = collections.Counter()

				# Loop through all time-phrase words
				for idx, time in enumerate(PL_TIME_WORDS):

					# Regular Expression for handling phrases of the form "from __ to __ minutes/hours"
					reg_to = re.compile('[1-9][0-9]*\sto\s[1-9][0-9]*\s{}'.format(time))

					# Loop through each matched time phrase in the list of instructions
					for matchString in reg_to.findall(instructions_lg):
						split = matchString.split(' ')
						timeA = int(split[0])
						timeB = int(split[2])

						# Get the average numeric time in the specified range
						times[time] += ((timeA + timeB) / 2.) * WEIGHTS[idx]

					# Regular Expression for handling phrases of the form "__ minutes/hours"
					reg = re.compile('(?<!\sto)\s[1-9][0-9]*\s{}'.format(time))

					# Loop through each matched time phrase in the list of instructions
					for matchString in reg.findall(instructions_lg):
						split = matchString.split(' ')
						timeA = int(split[1])

						# Get the numeric time specified
						times[time] += timeA * WEIGHTS[idx]

					# Regular Expression for handling phrases of the form "couple of minutes/hours"
					reg = re.compile('couple\sof\s{}'.format(time))

					# Get the total numeric time specified -- assume a "couple" minutes means 2 minutes
					times[time] += len(reg.findall(instructions_lg))*2*WEIGHTS[idx]

					# Regular Expression for handling phrases of the form "few minutes/hours"
					reg = re.compile('few\s{}'.format(time))

					# Get the total numeric time specified -- assume a "few" minutes means ~4 minutes
					times[time] += len(reg.findall(instructions_lg))*4*WEIGHTS[idx]

				# Save the time data for this recipe as value for key=recipe_id in larger dict {recipe id -> {time_word -> count}}
				time_map[recipe_id] = times

	# Write the full processed JSON data to its permanent location processed/instruction_time.json
	filename = 'processed/instruction_time.json'
	with open(filename, 'w') as fp:
		json.dump(time_map, fp, sort_keys=True, indent=4, separators=(',', ': '))

if __name__ == '__main__':
	main()