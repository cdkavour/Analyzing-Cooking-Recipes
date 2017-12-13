# EECS-595-Final-Project
Tool for analyzing cooking recipes to estimate and predict preparation time.

** Important Notes**
python parse_ingredients.py recipes --- this script will not run unless you have the NYTimes tagger installed on your machine

Python Files:
	extra_functions.py
	extract_features.py
	get_recipe_information.py
	model.py
	parse_ingredients.py
	parse_instruction_minutes.py
	parse_instructions.py
	parse_num_ingredients.py
	parse_num_instructions.py
	parse_times.py
	scraper_get_categories.py
	scraper_get_recipes.py
	verify_unique_recipes.py

Data Folders:
	recipes/ - stores json files of recipe data
	results/ - stores graphs/figures/outputs from our results
	links/ - stores recipe URLs scraped from allrecipes.com
	processed/ - stores processed json data for recipe data input to our model
	tmp/ - used for temporary storage thoughout the program
	ex_recipes/ - contains sample recipe json files for testing

How To Use:
The order of execution of the files is as follows:
1) python scraper_get_categories.py
2) pyton scraper_get_recipes.py
3) python get_recipe_information.py links
4) python parse_*.py
5) python model.py

** Important **
Steps 1-4 are all for the sake of collecting well formed data, in the form of JSON, that is used as input to the model.  The
fully correct data for the processed folder will be in processed.zip; you can extract this data to over-write the processed folder
and run python model.py on this well formed data to test the program.

The following describes the order of events of the program:

1) python scraper_get_categories.py:
This scrapes allrecipes.com to get the URLs for each recipe category page. From these pages,
we can scrape all the recipe URLs we need.  The URLs for each category page are stored in Category_urls.txt

2) pyton scraper_get_recipes.py
This scrapes each category page from Category_urls.txt for all the recipes on that page. It stores the recipe
URLs in the links/ directory, under a diffent number label i (links_i) for each of the i categories.

3) python get_recipe_information.py links
This scrapes each of the recipe URLs in a given links_i file (in which i is specified as a global variable in
the script), and gets all the relevant recipe information for each recipe, storing this info in json files in the recipes/
directory as recipes/recipes_i.json

4) python parse_ingredients.py recipes
python parse_instructions.py recipes
python parse_instruction_minutes.py recipes
python parse_num_ingredients.py recipes
python parse_num_instructions.py recipes
python parse_times.py recipes

Each of these scripts parses the recipe JSON data and produces a JSON file for the specific feature specified by the script's name.
Each JSON file produced is put in the processed/ directory, from where the data can be directly accessed as input data to our model.

5) python model.py
This script runs our machine learning model on the processed JSON recipe information as input, and produces our figures and
prints the accuracies of the model.
