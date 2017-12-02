# Parse allrecipes.com to get all categories.
# Ingredients category manually removed

from urllib.request import urlopen
from bs4 import BeautifulSoup

def main():
    urls = open("Category_urls.txt").read().splitlines()
    output = open("Recipe_urls.txt", 'w')
    recipe_urls = set()
    for base_url in urls:
        i = 1
        print(base_url)
        # Create soup for each category page
        url = base_url + '?page=' + str(i) + '#' + str(i)
        bytes = urlopen(url).read()
        soup = BeautifulSoup(bytes, 'lxml')
        for recipe in soup.find_all('article', 'fixed-recipe-card'):
            x = recipe.find('a', 'fixed-recipe-card__title-link')
            if x:
                recipe_urls.add(x.get('href'))
    for recipe in recipe_urls:
        output.write('http://allrecipes.com' + recipe + '\n')

if __name__ == '__main__':
    main()