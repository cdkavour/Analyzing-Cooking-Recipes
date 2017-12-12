'''
extra_functions.py

Assortment of additional miscellanious functions for use in data collection/parsing
'''

import json
import os, sys

# Returns the contents of a JSON file as a Dictionary object
def json_to_dict(filename):
	data = json.load(open(filename))
	return data