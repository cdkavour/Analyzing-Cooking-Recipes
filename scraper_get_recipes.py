# Parse allrecipes.com to get all categories.
# Ingredients category manually removed

from urllib.request import urlopen
from bs4 import BeautifulSoup
from collections import defaultdict

def main():
    URLS = open("Category_urls.txt").read().splitlines()
    count = 0
    for line in URLS:
        count += 1
        recipe_tags = defaultdict(list)

        line = line.split()
        print(line)
        page_found = True
        miss = 0
        i = 1
        while page_found:
            print(i)
            # Create soup for each category page
            url = line[0] + '?page=' + str(i) + '#' + str(i)
            try:
                bytes = urlopen(url).read()
                soup = BeautifulSoup(bytes, 'lxml')
            except:
                print("exception")
                miss += 1
                if miss >= 3:
                    page_found = False
                continue
            page_found = False
            for recipe in soup.find_all('article', 'fixed-recipe-card'):
                x = recipe.find('a', 'fixed-recipe-card__title-link')
                if x:
                    recipe_tags[x.get('href')].append(line[1])
                    page_found = True
            i += 1

        output = open("recipe_urls/Recipe_urls_" + str(count) + ".txt", 'w')
        for recipe in recipe_tags.keys():
            recipe_out = 'http://allrecipes.com' + recipe
            for tag in recipe_tags[recipe]:
                recipe_out += ' ' + tag
            print(recipe_out)
            output.write(recipe_out + '\n')
        output.close()

if __name__ == '__main__':
    main()