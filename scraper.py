# Parse Recipes from a given genre

from urllib.request import urlopen
from bs4 import BeautifulSoup

bbq_url = 'http://allrecipes.com/recipes/88/bbq-grilling/?page=8'

def main():
	# Create Soup for BBQ recipes page
	bytes = urlopen(bbq_url).read()
	soup = BeautifulSoup(bytes)

	# Loop over the recipe cards, and get the url for that recipe
	recipe_urls = []
	for recipe_card in soup.find_all('article', 'fixed-recipe-card'):
		if recipe_card.a:
			recipe_urls.append(recipe_card.a.get('href'))

	print(recipe_urls)
	print("Done.")

if __name__ == '__main__':
	main()