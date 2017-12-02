# Parse allrecipes.com to get all categories.
# Ingredients category manually removed

from urllib.request import urlopen
from bs4 import BeautifulSoup
from collections import defaultdict

def main():
    URLS = open("Category_urls.txt").read().splitlines()
    output = open("Recipe_urls.txt", 'w')
    recipe_tags = defaultdict(list)
    for line in URLS:
        line = line.split()
        i = 1
        print(line)
        # Create soup for each category page
        url = line[0] + '?page=' + str(i) + '#' + str(i)
        bytes = urlopen(url).read()
        soup = BeautifulSoup(bytes, 'lxml')
        for recipe in soup.find_all('article', 'fixed-recipe-card'):
            x = recipe.find('a', 'fixed-recipe-card__title-link')
            if x:
                recipe_tags[x.get('href')].append(line[1])
    for recipe in recipe_tags.keys():
        recipe_out = 'http://allrecipes.com' + recipe
        for tag in recipe_tags[recipe]:
            recipe_out += ' ' + tag
        print(recipe_out)
        output.write(recipe_out + '\n')

if __name__ == '__main__':
    main()