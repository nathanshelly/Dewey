import math, os, re, string, timeit, numpy, scipy.stats, random, shutil
from collections import defaultdict
from nltk import word_tokenize
from pickling import *

keys_to_ignore = ['total_words', 'num_files', 'mean_book_length', 'std_book_lenth']

##################################### Training

def generate_numeric_catalog(folder_path, file_name_list, des_genres):
    """ Generate dictionaries with frequency for each word in our training set. """
    # NOTE: relies on books_genres.p
    catalogs = {}
    books_genres = load("books_genres.p")

    if not file_name_list:
        file_name_list = [f for f in os.listdir(folder_path) if f != '.DS_Store']

    for file_name in file_name_list:
        genres = [genre for genre in books_genres[file_name] if genre in des_genres]
        if not genres: # this book doesn't match our genres
            continue

        # initialize catalog if none created yet
        for genre in genres:
            if genre not in catalogs:
                catalogs[genre] = defaultdict(int)
                catalogs[genre]['total_words'] = 0
                catalogs[genre]['num_files'] = 0
                catalogs[genre]['book_lengths'] = []

        count_occurrence_of_words(folder_path + file_name, catalogs, genres)

    # track book lengths
    # NOTE: did not end up being an informative feature
    for genre in catalogs.keys():
        catalogs[genre]['mean_book_length'] = numpy.mean(catalogs[genre]['book_lengths'])
        catalogs[genre]['std_book_lenth'] = numpy.std(catalogs[genre]['book_lengths'])
        del catalogs[genre]["book_lengths"]

    return catalogs

def count_occurrence_of_words(file_path, catalogs, genres):
    """ Count the number of occurences of a word, and store in the catalogs for the genres it belogns to. """
    words = word_tokenize(loadFile(file_path))

    for genre in genres:
        catalogs[genre]['num_files'] += 1
        catalogs[genre]["book_lengths"].append(len(words))

    for word in words:
        for genre in genres:
            catalogs[genre]['total_words'] += 1
            catalogs[genre][word.lower()] += 1

def generate_percentile_catalog(catalog):
    """ Convert one genre's feature dictionary from numeric to log frequency. """
    global keys_to_ignore
    total_words = catalog['total_words']
    num_files = catalog['num_files']
    mean_book_length = catalog['mean_book_length']
    std_book_lenth = catalog['std_book_lenth']
    perc_catalog = {'total_words': total_words, 'num_files': num_files, 'mean_book_length': mean_book_length, "std_book_lenth": std_book_lenth}

    for key, value in catalog.iteritems():
        if key not in keys_to_ignore:
            perc_catalog[key] = math.log(1.0*value/total_words)
    return perc_catalog

##################################### Smoothing

def smooth(dict_of_catalogs, word_list, smoothing_factor = 1):
    """ Takes a set of catalogs, and smooths each of them by adding the specified smoothing factor. """
    global keys_to_ignore
    for genre in dict_of_catalogs.keys():
        for word in word_list:
            if word in keys_to_ignore:
                continue
            dict_of_catalogs[genre][word] += smoothing_factor
            dict_of_catalogs[genre]['total_words'] += smoothing_factor
    return dict_of_catalogs

def word_list(dict_of_catalogs):
    """ Takes a set of catalogs and finds the set union of the features it contains (the words in all genres). """
    words = set()
    for genre_catalog in dict_of_catalogs.values():
        genre_words = set(genre_catalog.keys())
        words = words.union(genre_words)

    return words

##################################### Classification

def classify_text(string_to_classify, dict_of_catalogs, num_genres):
    """ Produces a sorted list of the num_genres most likely genres for a given text. """
    probs_dict = {key: 0 for key in dict_of_catalogs.keys()}
    compute_probabilites(probs_dict, dict_of_catalogs, word_tokenize(string_to_classify))
    return sorted(probs_dict, key=probs_dict.get, reverse=True)[:num_genres], sorted(probs_dict.values(), reverse=True)

def compute_probabilites(probs_dict, dict_of_catalogs, words_to_classify):
    """ Computes the probabilities of each genre given a text. """
    for key in dict_of_catalogs.keys():
        mean = dict_of_catalogs[key]['mean_book_length']
        std = dict_of_catalogs[key]['std_book_lenth']
        try:
            probs_dict[key] += scipy.stats.norm(mean, std).logpdf(len(words_to_classify))
        except ValueError:
            pass

    for word in words_to_classify:
        word = word.lower()
        for key in dict_of_catalogs.keys():
            try:
                probs_dict[key] += dict_of_catalogs[key][word]
            except KeyError:
                pass

##################################### Evaluation

def cross_validate(folds, books_path, smoothing_factor, genres):
    """ Perform k-fold cross-validation. """
    if not genres:
        genres = ['Adventure', 'Fantasy', 'Historical', 'Horror', 'Humor', 'Literature', 'Mystery', 'New_Adult', 'Other', 'Romance', 'Science_fiction', 'Teen', 'Themes', 'Thriller', 'Vampires', 'Young_Adult']

    percent = 1.0/folds

    books = os.listdir(books_path)
    random.shuffle(books)

    metrics = []

    for i in range(folds):
        print "FOLD " + str(i)

        # SPLIT UP DATA
        books_test = books[int(i*percent*len(books)):int((i+1)*percent*len(books))]
        books_train = list(set(books) - set(books_test))

        # TRAIN
        train_catalogs = generate_numeric_catalog(books_path, books_train, genres)
        train_catalogs = smooth(train_catalogs, word_list(train_catalogs), smoothing_factor)
        train_catalogs = {genre:generate_percentile_catalog(catalog) for genre, catalog in train_catalogs.iteritems()}

        # TEST
        twm = test(train_catalogs, books_test, books_path, genres)

        metrics.append(twm)
        print "metrics for this fold:", twm

    macroaverages = {}
    for genre in metrics[0].keys():
        macroaverages[genre] = {
            'precision': numpy.mean([acc[genre]['precision'] for acc in metrics]),
            'recall': numpy.mean([acc[genre]['recall'] for acc in metrics]),
            'F-measure': numpy.mean([acc[genre]['F-measure'] for acc in metrics])
            }
    return macroaverages, metrics

def test(train_catalogs, test_files, books_path, des_genres):
    """ Produce a set of precision, recall, and F-measures for each genre """
    # NOTE: relies on books_genres.p
    metrics = {genre:{'correct':0, 'classified_as':0, 'in_genre':0} for genre in des_genres}

    books_genres = load('books_genres.p')

    for f in test_files:
        book = loadFile(books_path + f)
        actual_cats = [genre for genre in books_genres[f] if genre in des_genres]
        classified_cats = classify_text(book, train_catalogs, len(actual_cats))
        in_common = set(actual_cats).intersection(set(classified_cats))
        for genre in actual_cats:
            metrics[genre]['in_genre'] += 1
        for genre in classified_cats:
            metrics[genre]['classified_as'] += 1
        for genre in in_common:
            metrics[genre]['correct'] += 1

    for genre in metrics.keys():
        try:
            metrics[genre]['precision'] = 1.0*metrics[genre]['correct']/metrics[genre]['classified_as']
        except ArithmeticError:
            metrics[genre]['precision'] = 0
            print "Precision for " + genre + " is zero."
        try:
            metrics[genre]['recall'] = 1.0*metrics[genre]['correct']/metrics[genre]['in_genre']
        except ArithmeticError:
            metrics[genre]['recall'] = 0
            print "Recall for " + genre + " is zero."
        try:
            metrics[genre]['F-measure'] = 2.0*metrics[genre]['precision']*metrics[genre]['recall']/(metrics[genre]['precision'] + metrics[genre]['recall'])
        except ArithmeticError:
            metrics[genre]['F-measure'] = 0
            print "Precision and recall for " + genre + " are both zero."

    return metrics
