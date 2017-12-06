from urllib2 import urlopen
from bs4 import BeautifulSoup
from collections import defaultdict
import json
import time
import sys, os


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
    
    i = '76'
    jsonDir = sys.argv[1]
    #for filename in os.listdir(jsonDir):
    filename = 'links_{}'.format(i)

    print('Processing link file {}'.format(i))

    URLS = open(os.path.join(jsonDir, filename)).read().splitlines()

    recipes = defaultdict(dict)

    output = open('recipes/recipes_{}.json'.format(i), 'w')

    # TODO -- Verify Uniqueness across link files
    totLen = len(URLS)
    curUrl = 0.
    for idx, line in enumerate(URLS):

        if (curUrl % 6) == 0:
            print('\t {}%'.format((curUrl/totLen) * 100))

        curUrl += 1

        r = Recipe()

        # Make soup for this recipe
        line = line.split()
        url = line[0]
        time.sleep(0)
        try:
            bytes = urlopen(url).read()
            soup = BeautifulSoup(bytes, 'lxml')
        except:
            print(':(')
            continue

        # Get ID
        id = ''.join(ch for ch in url if ch.isdigit())
        r.id = id

        # Get Tags
        for tag in line[1:]:
            r.tags.append(tag)

        # Get ingredients
        all_ingredients = soup.find_all('ul', 'dropdownwrapper')
        for ingredients in all_ingredients:
            for ingredient in ingredients.find_all('span', 'recipe-ingred_txt added'):
                r.ingredients.append(ingredient.string)

        # Get Instructions
        instructions = soup.find('ol', 'list-numbers recipe-directions__list')
        for instruction in instructions.find_all('span', 'recipe-directions__list--item'):
            r.instructions.append(instruction.string)

        # Get Ready In Time
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
        if r.ready == None or r.ready == 0:
            continue
        recipes[id] = r.to_dict()

    output.write(json.dumps(recipes, sort_keys=True, indent=3))

    output.close()

    #i += 1

if __name__ == '__main__':
    main()
