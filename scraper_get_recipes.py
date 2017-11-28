# Parse allrecipes.com to get all categories.
# Ingredients category manually removed

from urllib.request import urlopen
from bs4 import BeautifulSoup



def main():
	urls = open("Category_urls.txt").read().splitlines()

	for base_url in urls:
		i = 1
		while True:
			# Create soup for each category page
			url = base_url + '?page=' + str(i) + '#' + str(i)
			bytes = urlopen(url).read()
			soup = BeautifulSoup(bytes, 'lxml')
			for recipe in soup.find_all('article', 'fixed-recipe-card'):
				print(recipe.find('a', 'ng-isolate-scope'), '\n\n')
			# # Loop over the categories and snag the url they point to
			# category_urls = []
			# for category in soup.find_all('a', href=True, class_="hero-link__item"):
			# 	href = category['href']
			# 	if href[0] == '/':
			# 		category_urls.append('allrecipes.com' + href + '\n')
			# with open("Category_urls.txt", 'w') as output:
			# 	for URL in category_urls:
			# 		if "ingredients" in URL:
			# 			continue
			# 		output.write(URL)
			break
		break

if __name__ == '__main__':
	main()