"""
Usage:
	python parse_instructions.py DIR

	DIR - a directory containing the scraped data from allrecipes in JSON format (get_recipe_information.py)

Function:
	Generates processed/instructions.json, which maps recipeID to counts of the imperatives in its instructions

"""

import os, sys
import json
import nltk
import collections
from sets import Set
import re

#Returns map from recipeIDs -> imperatives -> counts
def get_instruction_time_words(jsonDir):

	PL_TIME_WORDS = ['minutes', 'hours']
	WEIGHTS = [1, 1]

	time_map = {}

	for filename in os.listdir(jsonDir):
		if filename.endswith('.json'):
			data = json.load(open(os.path.join(jsonDir, filename)))

			for recipe_id, recipe in data.iteritems():
				instructions_lg = ' '.join(recipe['instructions'])

				times = collections.Counter()

				idx = 0
				for time in PL_TIME_WORDS:

					# N to N TIME
					reg_to = re.compile('[1-9][0-9]*\sto\s[1-9][0-9]*\s{}'.format(time))
					for matchString in reg_to.findall(instructions_lg):
						split = matchString.split(' ')
						timeA = int(split[0])
						timeB = int(split[2])

						times[time] += ((timeA + timeB) / 2.) * WEIGHTS[idx]

					# N TIME
					reg = re.compile('(?<!\sto)\s[1-9][0-9]*\s{}'.format(time))
					for matchString in reg.findall(instructions_lg):
						split = matchString.split(' ')
						timeA = int(split[1])
						times[time] += timeA * WEIGHTS[idx]

					# couple of TIME
					reg = re.compile('couple\sof\s{}'.format(time))
					times[time] += len(reg.findall(instructions_lg))*2*WEIGHTS[idx]

					# few TIME
					reg = re.compile('few\s{}'.format(time))
					times[time] += len(reg.findall(instructions_lg))*4*WEIGHTS[idx]

					idx += 1

				time_map[recipe_id] = times

	filename = 'processed/instruction_time.json'
	with open(filename, 'w') as fp:
		json.dump(time_map, fp, sort_keys=True, indent=4, separators=(',', ': '))


#map uniqueid -> cooktime
#def get_labels(jsonDir):

if __name__ == '__main__':
	inDir = sys.argv[1]
	get_instruction_time_words(inDir)