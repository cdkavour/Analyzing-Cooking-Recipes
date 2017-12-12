'''
get_recipe_information.py

Module Description: Main function parses a list of URL links to recipes on allrecipes.com,
                    and creates a JSON file representing those recipes and all data
                    associated with them in json format. JSON file for the ith list of 
                    link is stored in the recipes/ directory as recipes_i.json

Classes:
    Recipe: Stores relevant scraped information about a given recipe
            Overloaded function __str__ for convinient printing of recipe info
            The recipe.to_dict() function allows for easy creation of a dictionary
                object from the recipe's info
'''

from urllib2 import urlopen
from bs4 import BeautifulSoup
from collections import defaultdict
import json
import time
import sys, os

# Global variable i identifies which set of URL links we want to scrape - CHANGE THIS AS NEEDED
i = '51'

class Recipe:
    def __init__(self):
        self.id = None
        self.tags = []
        self.ingredients = []
        self.instructions = []
        self.prep = None
        self.cook = None
        self.ready = None

    def __str__(self):
        string = '{\n'
        # string += '\tid: ' + str(self.id) + '\n'
        string += '\ttags: ' + str(self.tags) + '\n'
        string += '\tingredients: ' + str(self.ingredients) + '\n'
        string += '\tinstructions: ' + str(self.instructions) + '\n'
        string += '\tprep: ' + str(self.prep) + '\n'
        string += '\tcook: ' + str(self.cook) + '\n'
        string += '\tready: ' + str(self.ready) + '\n' + '}'
        return string

    def to_dict(self):
        r = dict()
        r['id'] = self.id
        r['tags'] = self.tags
        r['ingredients'] = self.ingredients
        r['instructions'] = self.instructions
        r['ready'] = self.ready
        return r


def main():
    print('Processing link file {}'.format(i))

    # Argument to the script is the name of the directory containing the URL links to be parsed
    jsonDir = sys.argv[1]
    filename = 'links_{}'.format(i)

    # Get list of URL links
    URLS = open(os.path.join(jsonDir, filename)).read().splitlines()

    # Initialize empty dictionary for storing recipe information
    recipes = defaultdict(dict)

    # Create and open our output file for writing to
    output = open('recipes/recipes_{}.json'.format(i), 'w')

    # Loop though the URLs; scrape each from allrecipes.com using Beautiful Soup
    for url_idx, line in enumerate(URLS):

        # Print out % of URLs parsed every once in a while as a sanity check
        if (url_idx % 6) == 0:
            print('\t {}%'.format((curUrl/len(URLS)) * 100))

        # Create Recipe object r for our current recipe
        r = Recipe()

        # Make Soup object for this recipe, for scraping data
        line = line.split()
        url = line[0]
        time.sleep(0)
        try:
            bytes = urlopen(url).read()
            soup = BeautifulSoup(bytes, 'lxml')
        except:
            print('Error accessing URL -- may have been blocked from allrecipes.com')
            continue

        # Get ID for this recipe (directly form the URL)
        id = ''.join(ch for ch in url if ch.isdigit())
        r.id = id

        # Scrape the Ready In Time for this recipe
        times = soup.find_all('li', 'prepTime__item')
        for time_option in times:
            time_type = time_option.find('p', 'prepTime__item--type')
            if time_type and time_type.string == 'Ready In':
                T = time_option.find('time')['datetime'][2:]
                minutes = 0
                if 'Days' in T:
                    days = T.partition('Days')
                    minutes += 24 * 60 * int(days[0])
                    T = days[2]
                elif 'Day' in T:
                    days = T.partition('Day')
                    minutes += 24 * 60 * int(days[0])
                    T = days[2]
                if 'H' in T:
                    hours = T.partition('H')
                    minutes += 60 * int(hours[0])
                    T = hours[2]
                if 'M' in T:
                    minutes += int(T.partition('M')[0])
                r.ready = minutes

        # Continue if the Ready In Time is zero for this recipe
        if r.ready == None or r.ready == 0:
            continue

        # Get category tags assciated from this recipe (directly from the URL)
        for tag in line[1:]:
            r.tags.append(tag)

        # Scrape the ingredients for this recipe
        all_ingredients = soup.find_all('ul', 'dropdownwrapper')
        for ingredients in all_ingredients:
            for ingredient in ingredients.find_all('span', 'recipe-ingred_txt added'):
                r.ingredients.append(ingredient.string)

        # Scrape the instructions for this recipe
        instructions = soup.find('ol', 'list-numbers recipe-directions__list')
        for instruction in instructions.find_all('span', 'recipe-directions__list--item'):
            r.instructions.append(instruction.string)

        # Create dictionary object from the scraped recipe data
        recipes[id] = r.to_dict()

    # Write dictionary object for this recipe to our output JSON file
    output.write(json.dumps(recipes, sort_keys=True, indent=3))
    output.close()

if __name__ == '__main__':
    main()
