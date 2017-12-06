# Parse allrecipes.com to get all categories.
# Ingredients category manually removed

from urllib.request import urlopen
from bs4 import BeautifulSoup
from collections import defaultdict
import time

def main():
    URLS = open("Category_urls.txt").read().splitlines()
    count = 20
    for line in URLS:
        count += 1
        recipe_tags = defaultdict(list)

        line = line.split()
        print(line)
        page_found = True        # Get Ready In Time
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
        i = 1
        miss = 0
        while page_found:
            print(i)
            # Create soup for each category page
            url = line[0] + '?page=' + str(i) + '#' + str(i)
            try:
                bytes = urlopen(url).read()
                soup = BeautifulSoup(bytes, 'lxml')
                page_found = False
                for recipe in soup.find_all('article', 'fixed-recipe-card'):
                    x = recipe.find('a', 'fixed-recipe-card__title-link')
                    if x:
                        recipe_tags[x.get('href')].append(line[1])
                        page_found = True
                miss = 0
                i += 1
            except:
                print("exception")
                time.sleep(1)
                miss += 1
                if miss >= 3:
                    page_found = False

        output = open("recipe_urls/Recipe_urls_" + str(count) + ".txt", 'w')
        for recipe in recipe_tags.keys():
            recipe_out = 'http://allrecipes.com' + recipe
            for tag in recipe_tags[recipe]:
                recipe_out += ' ' + tag
            output.write(recipe_out + '\n')
        output.close()

if __name__ == '__main__':
    main()