# import nltk, re, os, copy, timeit
import re, os, copy, timeit
import cPickle as pickle
from feature_extractor import *

def inspectBooks(features, loadfile, savefile):
	booksPath = 'books-opened'
	catalog = unpickleFile(loadfile)
	for book in os.listdir(booksPath):
		if book == ".DS_Store":
			continue
		print book
		extractor = feature_extractor(booksPath + '/' + book, catalog[book])
		# featuresDict = extractor.word_extract(features)
		# featuresDict = extractor.file_extract(features)
		featuresDict = extractor.existing_extract(features)

		for key, value in featuresDict.iteritems():
			catalog[book][key] = value

	pickleSomething(catalog, savefile)

def pickleSomething(data, fileName):
	pickleFile = open(fileName, 'w')
	pickle.dump(data, pickleFile)
	pickleFile.close()

def unpickleFile(fileName):
	pickleFile = open(fileName, 'r')
	data = pickle.load(pickleFile)
	return data

def dictAllBooks(path):
	booksDict = {}
	for book in os.listdir(path):
		booksDict[book] = {}
		booksDict[book]['genre'] = []

	return booksDict

def categorizeBooks():
	catalog = dictAllBooks('books-opened')
	for genre in os.listdir('books'):
		for book in os.listdir('books/' + str(genre)):
			if genre not in catalog[book]['genre']:
				catalog[book]['genre'].append(genre)

	pickleSomething(catalog, 'books.p')

def generateGenresDict():
	genresDict = {}
	for genre in os.listdir('books'):
		genresDict[genre] = []
		for book in os.listdir('books/' + str(genre)):
				genresDict[genre].append(book)
	pickleSomething(genresDict, 'genres.p')

def countNumOfGenre():
	genres = unpickleFile('genres.p')
	sizeOfGenres = {}
	numBooks = 0
	for genre, books in genres.iteritems():
		numBooks += len(books)
		sizeOfGenres[genre] = len(books)

	print numBooks
	return sizeOfGenres

def rebuildDataFiles():
	categorizeBooks()
	generateGenresDict()

def joinPickles(pickleWithMoreUniqueFeatures, pickleWithLessUniqueFeatures):
	dictWithMore = unpickleFile(pickleWithMoreUniqueFeatures)
	dictWithLess = unpickleFile(pickleWithLessUniqueFeatures)
	dictWithMoreKeys = dictWithMore.keys()
	dictWithLessKeys = dictWithLess.keys()

	print dictWithLessKeys
	print dictWithMoreKeys

	separateLists = list(set(temp1) - set(temp2))

	print separateLists

	# for key, value in variable:
	# 	pass
