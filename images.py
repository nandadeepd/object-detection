import urllib.request
import pandas

getImage = lambda url, name : urllib.request.urlretrieve(url, str(name) + str('.jpg'))
images = pandas.read_csv('urls.txt', sep = " ", header = None)

def main():

	for index, row in images.iterrows():
		getImage(row[0], index)
		
if __name__ == '__main__':
	main()
