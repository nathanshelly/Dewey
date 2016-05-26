import math, os, re, string, timeit
from collections import defaultdict
from nltk import word_tokenize
from pickling import *

##################################### Training

def bulk_train(path = 'books', genre = '', genres_list = [], smooth_factor = 1, name_offset = ''):
    '''Trains the Naive Bayes Sentiment Classifier using unigrams'''
    catalogs = {}
    for genre in os.listdir(path):
        print genre
        save(generate_numeric_catalog(path + '/' + genre), genre + '.p')

def generate_numeric_catalog(folder_path, file_name_list = []):
    '''Generate dictionaries with frequency for each word in our training set.'''
    catalog = defaultdict(int)
    catalog['total_words'] = 0
    catalog['num_files'] = 0

    if file_name_list:
        for file_name in file_name_list:
            count_occurrence_of_grams(folder_path + '/' + file_name, catalog)
    else:
        for file_name in os.listdir(folder_path):
            count_occurrence_of_grams(folder_path + '/' + file_name, catalog)
    return catalog

def count_occurrence_of_grams(file_path, catalog):
    '''Count the number of occurences of a word.'''
    # Catalog must have total_words and num_files keys
    print file_path
    catalog['num_files'] += 1
    temp_file = open(file_path)
    for line in temp_file:
        words = word_tokenize(line)
        for word in words:
            catalog['total_words'] += 1
            catalog[word.lower()] += 1
    temp_file.close()

def generate_percentile_catalog(catalog):
    '''Convert our feature dictionaries from numeric to log frequency (log(percentiles))'''
    total_words = catalog['total_words']
    perc_catalog = {'total_words': total_words, 'num_files': catalog['num_files']}

    for key, value in catalog.iteritems():
        if key not in ['total_words', 'num_files']:
            perc_catalog[key] = math.log(1.0*value/total_words)

    return perc_catalog

##################################### Classification

def classify_file(file_path, dict_of_catalogs):
    '''Given a target file, this function returns the most likely genre to which the target file belongs (i.e. fantasy, horror).'''
    probs_dict = {key: 0 for key in dict_of_catalogs.keys()}
    temp_file = open(file_path)
    for line in temp_file:
        classify_string(probs_dict, dict_of_catalogs, word_tokenize(line))

    return max(probs_dict, key=probs_dict.get) # return the key corresponding to the max value# probs.keys()[ind]

def classify_text(string_to_classify, dict_of_catalogs):
    '''Given a target string, this function returns the most likely genre to which the target string belongs (i.e. fantasy, horror).'''
    probs_dict = {key: 0 for key in dict_of_catalogs.keys()}
    classify_string(probs_dict, dict_of_catalogs, word_tokenize(string_to_classify))
    return max(probs_dict, key=probs_dict.get) # return the key corresponding to the max value# probs.keys()[ind]

def classify_string(probs_dict, dict_of_catalogs, words_to_classify):
    '''Takes string and updates the probabilities dictionary'''
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

def cross_validate(folds, path, smooth_factor = 1):
    """ Perform k-fold cross-validation. """
    files = os.listdir(path)
    pos_files = [temp_file for temp_file in files if 'movies-5' in temp_file]
    neg_files = [temp_file for temp_file in files if 'movies-1' in temp_file]
    percent = 1.0/folds
    pos = [0, 0, 0]
    neg = [0, 0, 0]

    for i in range(folds):
        neg_test = neg_files[int(i*percent*len(neg_files)):int((i+1)*percent*len(neg_files))]
        neg_train  = list(set(neg_files) - set(neg_test))
        pos_test = pos_files[int(i*percent*len(pos_files)):int((i+1)*percent*len(pos_files))]
        pos_train  = list(set(pos_files) - set(pos_test))

        train(pos_train = pos_train, neg_train = neg_train, path = path, smooth_factor = smooth_factor, name_offset = 'cv')
        results = bulk_test(path, pos_test, neg_test)

        pos = [x + y for x, y in zip(pos, results['positive'])]
        neg = [x + y for x, y in zip(neg, results['negative'])]

    pos_averages = [val/folds for val in pos]
    neg_averages = [val/folds for val in neg]
    return pos_averages, neg_averages

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

def master_word_list(catalogs_path):
    '''Create list of all words in our library.'''
    words = set()
    for genre_pickle in os.listdir(catalogs_path):
        genre_words = set(load(catalogs_path + genre_pickle).keys())
        words = words.union(genre_words)

    save(words, 'all_words_list.p')
    return words

def smooth_all(catalogs_path, smoothed_path, master_word_list, smoothing_factor):
    for catalog in os.listdir(catalogs_path):
        cat = load(catalogs_path + catalog)
        for word in master_word_list:
            if word in ['num_files', 'total_words']:
                continue
            cat[word] += smoothing_factor
            cat['total_words'] += smoothing_factor

        save(cat, smoothed_path + catalog[:-2] + "_smoothed.p")

##################################### Main

def main():
    catalog_path = 'catalogs_smoothed/'
    smoothed_ending = '_smoothed.p'
    catalogs = {'Teen': generate_percentile_catalog(load(catalog_path + 'Teen' + smoothed_ending)), 'Horror': generate_percentile_catalog(load(catalog_path + 'Horror' + smoothed_ending))}
    bulk_test(catalogs)

main()
