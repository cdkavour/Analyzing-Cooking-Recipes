# Parse allrecipes.com to get all categories.
# Ingredients category manually removed

from urllib.request import urlopen
from bs4 import BeautifulSoup

url = 'http://allrecipes.com/recipes/?grouping=all'

def main():
	# Create Soup all categories
	bytes = urlopen(url).read()
	soup = BeautifulSoup(bytes, 'lxml')

	# Loop over the categories and snag the url they point to
	category_urls = []
	for category in soup.find_all('a', href=True, class_="hero-link__item"):
		href = category['href']
		if href[0] == '/':
			category_urls.append('http://allrecipes.com' + href + '\n')
	with open("Category_urls.txt", 'w') as output:
		for URL in category_urls:
			if "ingredients" in URL:
				continue
			output.write(URL)

if __name__ == '__main__':
	main()