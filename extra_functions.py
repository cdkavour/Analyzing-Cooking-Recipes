import json
import os, sys

def json_to_dict(filename):
	data = json.load(open(filename))
	return data