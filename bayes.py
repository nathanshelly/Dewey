import math, os, re, string, timeit, numpy, scipy.stats
from collections import defaultdict
from nltk import word_tokenize
from pickling import *

##################################### Training

def bulk_train(path = 'books', genre = '', smooth_factor = 1, name_offset = ''):
    """Trains the Naive Bayes Sentiment Classifier using unigrams"""
    catalogs = {}
    for genre in os.listdir(path):
        print genre
        save(generate_numeric_catalog(path + '/' + genre), genre + '.p')

def generate_numeric_catalog(folder_path, file_name_list = []):
    '''Generate dictionaries with frequency for each word in our training set.'''
    catalog = defaultdict(int)
    catalog['total_words'] = 0
    catalog['num_files'] = 0
    catalog['book_lengths'] = []

    if file_name_list:
        for file_name in file_name_list:
            count_occurrence_of_grams(folder_path + '/' + file_name, catalog)
    else:
        for file_name in os.listdir(folder_path):
            count_occurrence_of_grams(folder_path + '/' + file_name, catalog)

    catalog['mean_book_length'] = numpy.mean(catalog['book_lengths'])
    catalog['std_book_lenth'] = numpy.std(catalog['book_lengths'])
    del catalog["book_lengths"]

    return catalog

def count_occurrence_of_grams(file_path, catalog):
    '''Count the number of occurences of a word.'''
    # Catalog must have total_words and num_files keys
    # print file_path
    catalog['num_files'] += 1
    words = word_tokenize(loadFile(file_path))
    catalog["book_lengths"].append(len(words))
    for word in words:
        catalog['total_words'] += 1
        catalog[word.lower()] += 1

def generate_percentile_catalog(catalog):
    '''Convert our feature dictionaries from numeric to log frequency (log(percentiles))'''
    total_words = catalog['total_words']
    perc_catalog = {'total_words': total_words, 'num_files': catalog['num_files']}

    for key, value in catalog.iteritems():
        if key not in ['total_words', 'num_files']:
            perc_catalog[key] = math.log(1.0*value/total_words)

    return perc_catalog

##################################### Classification

def classify_text(string_to_classify, dict_of_catalogs):
    '''Given a target string, this function returns the most likely genre to which the target string belongs (i.e. fantasy, horror).'''
    probs_dict = {key: 0 for key in dict_of_catalogs.keys()}
    # print "probs_dict before", probs_dict
    update_probabilites(probs_dict, dict_of_catalogs, word_tokenize(string_to_classify))
    # print "probs_dict after", probs_dict
    return max(probs_dict, key=probs_dict.get) # return the key corresponding to the max value# probs.keys()[ind]

def update_probabilites(probs_dict, dict_of_catalogs, words_to_classify):
    '''Takes string and updates the probabilities dictionary'''

    for key in dict_of_catalogs.keys():
        mean = dict_of_catalogs[key]['mean_book_length']
        std = dict_of_catalogs[key]['std_book_lenth']
        probs_dict[key] += math.log(scipy.stats.norm(mean, std).pdf(len(words_to_classify)))

    # print words_to_classify
    for word in words_to_classify:
        word = word.lower()
        for key in dict_of_catalogs.keys():
            try:
                probs_dict[key] += dict_of_catalogs[key][word]
            except KeyError:
                pass

##################################### Evaluation

def evaluate(folds, path = 'movie_reviews/movies_reviews', smooth_factor = 1):
    '''Run k-fold cross-validation on data set, generating recall, precision and F-measure values for all classes'''
    pos_averages, neg_averages = cross_validate(folds, path, smooth_factor)
    for i in range(len(pos_averages)):
        pos_averages[i] = round(pos_averages[i], 6)
        neg_averages[i] = round(neg_averages[i], 6)
    print 'Pos - [recall, precision, F1 measure]', pos_averages
    print 'Neg - [recall, precision, F1 measure]', neg_averages

def test(train_catalogs, test_files, genre, books_path):
    correct = 0.0
    print "genre", genre
    for f in test_files:
        # print f
        book = loadFile(books_path + genre + '/' + f)
        cat = classify_text(book, train_catalogs)
        print "classified as", cat
        if cat == genre:
            correct += 1
    print genre, "correct", correct
    return correct/len(test_files)

def cross_validate(genres, folds, books_path, smoothing_factor):
    """ Perform k-fold cross-validation. """
    books = {}
    books_test = {}
    train_catalogs = {}
    percent = 1.0/folds

    for genre in genres:
        books[genre] = [f for f in os.listdir(books_path + genre)]
    # print "books", books

    accuracies = {genre:[] for genre in genres}

    for i in range(folds):
        print "FOLD " + str(i)
        print "Working on: "
        for genre in genres:
            print genre
            books_test[genre] = books[genre][int(i*percent*len(books[genre])):int((i+1)*percent*len(books[genre]))]

            # print "books[genre] ", books[genre]
            # print "books_test[" + genre + "] ", books_test[genre]
            books_train = list(set(books[genre]) - set(books_test[genre]))
            # print "test ", books_test
            # print "train set" + genre, books_train
            train_catalogs[genre] = generate_numeric_catalog(books_path + genre, books_train)
            # print "train_catalogs", train_catalogs

        # smooth the catalogs
        train_catalogs = smooth(train_catalogs, word_list(train_catalogs), smoothing_factor)
        train_catalogs = {genre:generate_percentile_catalog(catalog) for genre, catalog in train_catalogs.iteritems()}

        for genre in genres:
            # print "books_test", books_test
            # print "genre", genre
            # print "books_test[genre]", books_test[genre]
            accuracies[genre].append(test(train_catalogs, books_test[genre], genre, books_path))
        print "accuracies", accuracies

    results = {genre:math.fsum(accs)/folds for genre, accs in accuracies.iteritems()}
    return results, accuracies

def bulk_test(dict_of_catalogs, path = 'books'):
    '''Run class_test on all classes, generate recall, precision and F-Measure values for each class'''
    f = open('results.txt', 'w')
    temp_results = {}
    for classification in dict_of_catalogs.keys():
        print classification
        files_to_test = os.listdir(path + '/' + classification)[:len(os.listdir(path + '/' + classification))/4]
        print len(files_to_test)
        temp_results[classification] = class_test(path, classification, dict_of_catalogs, files_to_test)
        print classification + ': ', 1.0*temp_results[classification][0]/temp_results[classification][1]
        f.write(classification + ': ' + str(1.0*temp_results[classification][0]/temp_results[classification][1]))
        # temp_results = {classification: class_test(path, classification, dict_of_catalogs) for classification in dict_of_catalogs.keys()}

    overall = 1.0 * sum([x[0] for x in temp_results.values()]) / sum([x[1] for x in temp_results.values()])
    f.write(str(overall) + '\n\n')
    f.close()
    print overall
    return overall

def class_test(path, correct_klass, dict_of_catalogs, files_to_test=[]):
    '''Perform classification on a list of files, assumed to be of the same target class.'''
    path += '/'+correct_klass
    correct = 0
    total = 0
    if not files_to_test:
        files_to_test = os.listdir(path)
    for name in files_to_test:
        # print name
        start_time = timeit.default_timer()
        sentiment = classify_file(path + '/' + name, dict_of_catalogs)
        # sentiment = classify_text(loadFile(path + '/' + name), dict_of_catalogs)
        end_time = timeit.default_timer()
        print 'Classify time: ', end_time - start_time
        # print 'Name: ', name
        # print 'Correct classification: ', correct_klass
        # print 'Generated Sentiment: ', sentiment
        if sentiment == correct_klass:
            correct += 1
        total += 1

    return [correct, total]

##################################### Smoothing

def smooth(catalogs, word_list, smoothing_factor = 1):
    smooth_ignore_list = ['num_files', 'total_words', 'mean_book_length', "std_book_lenth"]

    for genre in catalogs.keys():
        for word in word_list:
            if word in smooth_ignore_list:
                continue
            catalogs[genre][word] += smoothing_factor
            catalogs[genre]['total_words'] += smoothing_factor
    return catalogs

def word_list(catalogs):
    words = set()
    for genre_catalog in catalogs.values():
        genre_words = set(genre_catalog.keys())
        words = words.union(genre_words)

    return words
