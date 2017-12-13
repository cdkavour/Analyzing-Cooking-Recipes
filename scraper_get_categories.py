'''
scraper_get_categories.py

Usage:
	python scraper_get_categories.py

Module function:
	Parse allrecipes.com to get all categories.
'''

from urllib2 import urlopen
from bs4 import BeautifulSoup

# Url for main page on allrecipes.com, which contains urls for category pages
# Each category page has the recipe URLs for recipes of that category
main_url = 'http://allrecipes.com/recipes/?grouping=all'

def main():
	# Create Soup over all categories
	bytes = urlopen(main_url).read()
	soup = BeautifulSoup(bytes, 'lxml')

	# Loop over the categories
	category_urls = []
	for category in soup.find_all('a', href=True, class_="hero-link__item"):

		# Get the URL from each 'a' tag in the html identified by the soup
		href = category['href']
		if href[0] == '/':

			# Get the href_tag, append it to the full url
			start_idx = href.rfind('/', 0, len(href) - 1) + 1
			end_idx = len(href) - 1
			href_tag = href[start_idx:end_idx]
			url = 'http://allrecipes.com' + href + ' ' + href_tag + '\n'

			# Append the URL
			category_urls.append(url)

	# Write list of category urls to Category_urls.txt file
	with open("Category_urls.txt", 'w') as output:
		for URL in category_urls:
			output.write(URL)

if __name__ == '__main__':
	main()
