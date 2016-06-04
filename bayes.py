import math, os, re, string, timeit, numpy, scipy.stats, random, shutil
from collections import defaultdict
from nltk import word_tokenize
from pickling import *

keys_to_ignore = ['total_words', 'num_files', 'mean_book_length', 'std_book_lenth']

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

def generate_numeric_catalog_multiple(folder_path, file_name_list = []):
    '''Generate dictionaries with frequency for each word in our training set.'''
    catalogs = {}
    books_genres = loadFile("books_genres.p")

    if not file_name_list:
        file_name_list = os.listdir(folder_path)

    for file_name in file_name_list:
        genres = books_genres[file_name]
        for genre in genres:
            if genre not in catalogs:
                catalogs[genre] = defaultdict(int)
                catalogs[genre]['total_words'] = 0
                catalogs[genre]['num_files'] = 0
                catalogs[genre]['book_lengths'] = []
        count_occurrence_of_grams_multiple(folder_path + '/' + file_name, catalogs, genres)

    for genre in catalogs.keys():
        catalogs[genre]['mean_book_length'] = numpy.mean(catalogs[genre]['book_lengths'])
        catalogs[genre]['std_book_lenth'] = numpy.std(catalogs[genre]['book_lengths'])
        del catalogs[genre]["book_lengths"]

    return catalogs


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

def count_occurrence_of_grams_multiple(file_path, catalogs, genres):
    '''Count the number of occurences of a word.'''
    # Catalogs holds {genre:catalog}
    # Catalog must have total_words and num_files keys
    words = word_tokenize(loadFile(file_path))

    for genre in genres:
        catalogs[genre]['num_files'] += 1
        catalogs[genre]["book_lengths"].append(len(words))

    for word in words:
        for genre in genres:
            catalogs[genre]['total_words'] += 1
            catalogs[genre][word.lower()] += 1


def add_features():
    for numeric_catalog_path in os.listdir('catalogs'):
        if numeric_catalog_path == '.DS_Store':
            continue
        genre_name = numeric_catalog_path[:-2]
        numeric_catalog_path = "catalogs/" + numeric_catalog_path
        numeric_catalog = load(numeric_catalog_path)
        numeric_catalog['book_lengths'] = []
        for book_path in os.listdir('books/'+genre_name):
            numeric_catalog['book_lengths'].append(len(word_tokenize(loadFile('books/'+genre_name+'/'+book_path))))
        numeric_catalog['mean_book_length'] = numpy.mean(numeric_catalog['book_lengths'])
        numeric_catalog['std_book_lenth'] = numpy.std(numeric_catalog['book_lengths'])
        del numeric_catalog["book_lengths"]
        save(numeric_catalog, numeric_catalog_path)

def generate_percentile_catalog(catalog):
    '''Convert our feature dictionaries from numeric to log frequency (log(percentiles))'''
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

def split_genres():
    single_genre_folder = 'single_genre/'
    if not os.path.exists(single_genre_folder):
        os.makedirs(single_genre_folder)
    books_genres = load('books_genres.p')
    for folder in os.listdir('books'):
        if folder == ".DS_Store":
            continue
        if not os.path.exists(single_genre_folder + folder):
            os.makedirs(single_genre_folder + folder)
        for book in os.listdir('books/' + folder):
            if len(books_genres[book]) == 1:
                shutil.copy2('books/' + folder + '/' + book, single_genre_folder + folder + '/' + book)

    genres = {}
    for folder in os.listdir(single_genre_folder):
        if folder == ".DS_Store":
            continue
        genres[folder] = len(os.listdir(single_genre_folder + folder))
    sorted_genres = sorted(genres, key=genres.get, reverse=True)
    for folder in sorted_genres[5:]:
        if folder == ".DS_Store":
            continue
        shutil.rmtree(single_genre_folder + folder)

##################################### Smoothing

def smooth(dict_of_catalogs, word_list, smoothing_factor = 1):
    global keys_to_ignore
    for genre in dict_of_catalogs.keys():
        for word in word_list:
            if word in keys_to_ignore:
                continue
            dict_of_catalogs[genre][word] += smoothing_factor
            dict_of_catalogs[genre]['total_words'] += smoothing_factor
    return dict_of_catalogs

def word_list(dict_of_catalogs):
    words = set()
    for genre_catalog in dict_of_catalogs.values():
        genre_words = set(genre_catalog.keys())
        words = words.union(genre_words)

    return words

##################################### Classification

# for only one genre per books
def classify_text(string_to_classify, dict_of_catalogs):
    '''Given a target string, this function returns the most likely genre to which the target string belongs (i.e. fantasy, horror).'''
    probs_dict = {key: 0 for key in dict_of_catalogs.keys()}
    update_probabilites(probs_dict, dict_of_catalogs, word_tokenize(string_to_classify))
    return max(probs_dict, key=probs_dict.get) # return the key corresponding to the max value probs.keys()[ind]

# for multiple genres per book
def classify_text_multiple(string_to_classify, dict_of_catalogs, num_genres):
    probs_dict = {key: 0 for key in dict_of_catalogs.keys()}
    update_probabilites(probs_dict, dict_of_catalogs, word_tokenize(string_to_classify))
    return sorted(probs_dict, key=probs_dict.get, reverse=True)[:num_genres]

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

def cross_validate(genres, folds, books_path, smoothing_factor):
    """ Perform k-fold cross-validation. """
    books = {}
    books_test = {}
    train_catalogs = {}
    percent = 1.0/folds

    for genre in genres:
        filenames = os.listdir(books_path + genre)
        random.shuffle(filenames)
        books[genre] = [f for f in filenames]
    # print "books", books

    accuracies = {genre:[] for genre in genres}

    for i in range(folds):
        print "FOLD " + str(i)
        print "Working on: "
        for genre in genres:
            print genre
            books[genre] = books[genre]
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
        # print train_catalogs

        for genre in genres:
            # print "books_test", books_test
            # print "genre", genre
            # print "books_test[genre]", books_test[genre]
            accuracies[genre].append(test(train_catalogs, books_test[genre], genre, books_path))
        print "accuracies", accuracies

    results = {genre:math.fsum(accs)/folds for genre, accs in accuracies.iteritems()}
    return results, accuracies

# For multiple genres per book
def cross_validate_multiple(folds, books_path, smoothing_factor):
    """ Perform k-fold cross-validation. """
    percent = 1.0/folds

    books = os.listdir(books_path)

    metrics = []

    for i in range(folds):
        print "FOLD " + str(i)

        books_test = books[int(i*percent*len(books)):int((i+1)*percent*len(books))]
        books_train = list(set(books[genre]) - set(books_test[genre]))

        # NOTE: This line is going to have to be different
        train_catalogs = generate_numeric_catalog_multiple(books_path, books_train)

        # smooth the catalogs
        train_catalogs = smooth(train_catalogs, word_list(train_catalogs), smoothing_factor)
        train_catalogs = {genre:generate_percentile_catalog(catalog) for genre, catalog in train_catalogs.iteritems()}

        twm = test_with_measures(train_catalogs, books_test, books_path)
        accuracies.append(twm)
        print "metrics for this fold:", twm

    for genres in accuracies[0].keys():
        macroaverages[genre] = {
            'precision': math.fsum([acc[genre]['precision'] for acc in accuracies]),
            'recall': math.fsum([acc[genre]['recall'] for acc in accuracies]),
            'F-measure': math.fsum([acc[genre]['F-measure'] for acc in accuracies])
            }
    return macroaverages, accuracies

def bulk_test(dict_of_catalogs, divisor = 1, path = 'books/'):
    '''Run test on all given classes'''
    temp_results = {}
    for genre in dict_of_catalogs.keys():
        temp_files_list = os.listdir(path + genre)
        files_to_test = temp_files_list[:len(temp_files_list)/divisor]
        # print len(files_to_test)
        temp_results[genre] = test(dict_of_catalogs, files_to_test, genre, path)
        # print genre + ': ', temp_results[genre]

    f = open('smoothing_results.txt', 'a')
    f.write('Final Accuracies:\n')
    print
    print 'Final Accuracies: '
    for genre in temp_results.keys():
        print genre + ': ', temp_results[genre]
        f.write(str(genre) + ': ' + str(temp_results[genre]) + '\n')
    f.write('\n\n')
    print
    print
    f.close()

def test(train_catalogs, test_files, genre, books_path):
    ''' Calculate the accuracy of classification on a set of books, all from the same genre '''
    correct = 0.0
    print "genre:", genre
    if not test_files:
        test_files = os.listdir(books_path + genre)
    for f in test_files:
        # print f
        book = loadFile(books_path + genre + '/' + f)
        cat = classify_text(book, train_catalogs)
        # print "classified as", cat
        if cat == genre:
            correct += 1
    print genre, "correct:", correct
    smoothing_file = open('smoothing_results.txt', 'a')
    smoothing_file.write(str(genre) + ' correct: ' + str(correct) + '\n')
    smoothing_file.close()
    return correct/len(test_files)

# Assumes we have books_genres.p
def test_with_measures(train_catalogs, test_files, books_path):
    ''' Produce a set of precision, recall, and F-measures for each genre '''
    metrics = {genre:{'correct':0, 'classified_as':0, 'in_genre':0} for genre in train_catalogs.keys()}

    books_genres = loadFile('books_genres.p')

    for f in test_files:
        book = loadFile(books_path + f)
        actual_cats = books_genres[f]
        classified_cats = classify_text_multiple(book, train_catalogs, len(correct_cats))
        in_common = set(actual_cats).intersection(cats)
        for genre in actual_cats:
            metrics[genre]['in_genre'] += 1
        for genre in classified_cats:
            metrics[genre]['classified_as'] += 1
        for genre in in_common:
            metrics[genre]['correct'] += 1

    for genre in metrics.keys():
        metrics[genre]['precision'] = 1.0*metrics[genre]['correct']/metrics[genre]['classified_as']
        metrics[genre]['recall'] = 1.0*metrics[genre]['correct']/metrics['in_genre']
        metrics[genre]['F-measure'] = 2.0*metrics[genre]['precision']*metrics[genre]['recall']/(metrics[genre]['precision'] + metrics[genre]['recall'])

    return metrics
