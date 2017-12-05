'''
Verify That All Recipes across Links are unique
Print IDs of duplicate recipes
'''

def main():
	ids = []
	for i in ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']:
		links = open('links/links_{}'.format(i)).read().splitlines()
		ids.append([link.split('/')[4] for link in links])
	duplicates = [id for id in ids if ids.count(id) > 1]
	print('Link duplicates:\n{}'.format(duplicates))

if __name__ == '__main__':
	main()
