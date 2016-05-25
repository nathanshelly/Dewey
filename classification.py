# import nltk, re, os, copy
import re, os, copy
import cPickle as pickle
from subset_catalog import *
from pickling import *
from math import ceil

from sklearn import tree

def driver():
	'''Drives the process, gathering the catalog and making a classifier from the data'''

	"""
	catalog = unpickleFile('books.p')

	genres = ["Romance", "Horror"]
	features = ["cock_freq", "genre"]
	new_catalog = get_genres(catalog, genres, features)

	(train_set, test_set) = train_and_test(new_catalog, .2)
	duplicated_train_set = duplicate_genres(train_set)
	uniqueified_train_set = only_genres(duplicated_train_set, genres)

	format_train = format_catalog_for_classify(uniqueified_train_set)
	format_test = format_catalog_for_classify(test_set)

	print "train", format_train[:25]
	print "test", format_test[:25]

	NBClassifier = nltk.NaiveBayesClassifier.train(format_train)#, binary=False)
	pickleSomething(NBClassifier, 'tree.p')

	# print NBClassifier.pretty_format(depth=10000)
	print "Accuracy: ", test_classifier(format_test, NBClassifier)
	"""

	catalog = unpickleFile('books.p')

	genres = ["Literature", "Romance"]
	# features = ["vamp_freq", "genre"]
	features = ['unique_freq', 'genre']
	# features = None
	new_catalog = get_genres(catalog, genres, features)

	train, test = train_and_test(new_catalog, 0.2)

	print len(train), len(test)
	print

	dup_train = duplicate_genres(train)
	unique_train = only_genres(dup_train, genres)

	train_labels, train_features = reformat_catalog(unique_train)
	test_labels, test_features = reformat_catalog(test)

	# experimental code, don't keep

	train_features = [ [ nominalize(feature) for feature in feature_set ] for feature_set in train_features ]
	test_features = [ [ nominalize(feature) for feature in feature_set ] for feature_set in test_features ]

	print len(train_labels), len(test_labels)
	print

	print len(train_features), len(test_features)
	print

	print "train"
	# print train_labels[:25]
	print train_features[:25]
	print

	print "test"
	# print test_labels[:25]
	# print [x[1] for x in test_features][:50]
	# print [x[2] for x in test_features][:50]
	# print [x[3] for x in test_features][:50]
	# print [x[4] for x in test_features][:50]
	# print [x[5] for x in test_features][:50]

	print

	# dec_tree = tree.DecisionTreeClassifier(criterion="entropy", max_depth=1)
	dec_tree = tree.DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=20, max_depth=10)
	dec_tree = dec_tree.fit(train_features, train_labels)

	tree.export_graphviz(dec_tree, "vamp_tree.dot", filled=True, leaves_parallel=True)
	print "exported"

	pickleSomething(dec_tree, 'vamp_tree.p')
	print "pickled"

	predictions = dec_tree.predict(test_features)

	correct = 0
	for i in range(len(predictions)):
		if predictions[i] in test_labels[i]:
			correct += 1

	print "Accuracy: ", 1.0*correct/len(predictions)

def nominalize(feature_value):
	if feature_value > 1:
		return ceil((feature_value - 3) * 2)
	magnify = feature_value * 1000
	if magnify == 0:
		return 0
	elif magnify < .5:
		return 1
	elif magnify < 1:
		return 2
	elif magnify < 2:
		return 3
	else:
		return 4

def test_classifier(test_set, classifier):
	'''Takes a labeled test data set and returns the accuracy of the classifier on that set'''
	correct = 0
	for data_point in test_set:
		classify = classifier.classify(data_point[0])
		print classify, data_point
		if classify in data_point[1]:
			# print classify, data_point[1]
			# print "classified correctly"
			correct += 1
	return 1.0*correct/len(test_set)

driver()
