import math, os, re, string
import cPickle as pickle
from collections import defaultdict
from nltk import word_tokenize


##################################### Training

def train(path = 'books', pos_train = [], neg_train = [], smooth_factor = 1, name_offset = ''):
     """Trains the Naive Bayes Sentiment Classifier using unigrams"""
     if not pos_train or not neg_train:
         files = os.listdir(path)
         pos_train = [temp_file for temp_file in files if 'movies-5' in temp_file]
         neg_train = [temp_file for temp_file in files if 'movies-1' in temp_file]
     pos_catalog = generate_numeric_catalog(path, pos_train)
     neg_catalog = generate_numeric_catalog(path, neg_train)
     (neg_catalog, pos_catalog) = smooth(neg_catalog, pos_catalog, smooth_factor, name_offset)
     neg_catalog = generate_percentile_catalog(neg_catalog)
     pos_catalog = generate_percentile_catalog(pos_catalog)

     save(neg_catalog, 'neg_' + name_offset + '.p')
     save(pos_catalog, 'pos_' + name_offset + '.p')

     neg_features = neg_catalog
     pos_features = pos_catalog

def bulk_train(path = 'books', genre = '', smooth_factor = 1, name_offset = ''):
     """Trains the Naive Bayes Sentiment Classifier using unigrams"""
     catalogs = {}
     print os.listdir(path)[0:8]
    #  print os.listdir(path)[8:16]
     for genre in os.listdir(path)[0:8]:
          print genre
          save(generate_numeric_catalog(path + '/' + genre), genre + '.p')
          # catalogs[genre] = generate_percentile_catalog(generate_numeric_catalog(path + '/' + genre))
          # save(catalogs[genre], genre + '.p')
     # save(generate_percentile_catalog(generate_numeric_catalog(path + '/' + genre)), genre + '.p')

def generate_numeric_catalog(folder_path, file_name_list = []):
     """ Generate dictionaries with frequency for each word in our training set. """
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
     """ Count the number of occurences of a word. """
     # Catalog must have total_words and num_files keys
     print file_path
     file_string = loadFile(file_path)
     catalog['num_files'] += 1

     words = word_tokenize(file_string)
     for word in words:
         catalog['total_words'] += 1
         catalog[word.lower()] += 1

def smooth(first_catalog, second_catalog, num_to_add = 1):
     """ Perform add-one (add-num_to_add) smoothing on existing features dictionaries. """
     first_keys = set(first_catalog.keys())
     second_keys = set(second_catalog.keys())

     first_new = list(second_keys - first_keys)
     second_new = list(first_keys - second_keys)

     add_keys(first_catalog, first_new, num_to_add)
     add_keys(second_catalog, second_new, num_to_add)

     return (first_catalog, second_catalog)

def smooth_dict_of_catalogs(dict_of_catalogs, num_to_add, name_offset):
     """ Perform add-one (add-num_to_add) smoothing on existing features dictionaries. """
     negative_set_keys = set(neg_catalog.keys())
     positive_set_keys = set(pos_catalog.keys())

     neg_new = list(positive_set_keys - negative_set_keys)
     pos_new = list(negative_set_keys - positive_set_keys)

     add_keys(neg_catalog, neg_new, num_to_add)
     add_keys(pos_catalog, pos_new, num_to_add)

     return (neg_catalog, pos_catalog)

def generate_percentile_catalog(catalog):
     """ Convert our feature dictionaries from numeric to log frequency (log(percentiles)) """
     total_words = catalog['total_words']
     perc_catalog = {'total_words': total_words, 'num_files': catalog['num_files']}

     for key, value in catalog.iteritems():
         if key not in ['total_words', 'num_files']:
             perc_catalog[key] = math.log(1.0*value/total_words)

     return perc_catalog

def add_keys(catalog, keys, num):
     """ Add a list of keys to a dictionary. """
     for key in keys:
         catalog[key] = 0
     add_num(catalog, num)

def add_num(catalog, num):
     """ Add a number to each item in a dictionary (for smoothing). """
     for key, val in catalog.iteritems():
         if key not in ['total_words', 'num_files']:
             catalog[key] += num
             catalog['total_words'] += num

##################################### Classification

def classify(sText, dict_of_catalogs):
     """Given a target string sText, this function returns the most likely document
     class to which the target string belongs (i.e., positive, negative or neutral).
     """
     words = word_tokenize(sText)

     probs = {}
     for key in dict_of_catalogs.keys():
          probs[key] = 0
     # probs = {key: 0 for key in dict_of_catalogs.keys()}

     for word in words:
         word = word.lower()
         for key in dict_of_catalogs.keys():
             try:
                 probs[key] += dict_of_catalogs[key][word]
             except KeyError:
                 pass

     # if pick_neutral(probs, neutral_thresh):
     #     print "neutral"
     #     return 'neutral'

     # m = max(probs.values())
     # ind = [i for i, j in enumerate(probs.values()) if j == m][0]
     return max(probs, key=probs.get) # return the key corresponding to the max value# probs.keys()[ind]

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

def bulk_test(path, dict_of_catalogs):
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

     # temp_results['positive'] = class_test(path, 'positive', pos_test)
     # temp_results['negative'] = class_test(path, 'negative', neg_test)

     # results = {  }
     # keys = temp_results.keys()

     # for key, vals_list in temp_results.iteritems():
     #     results_list = [1.0*vals_list[0]/vals_list[1]]
         # print 'Recall for ' + key + ': ', results_list[0]

         # if key == 'positive':
         #     false_positives = temp_results['negative'][1] - temp_results['negative'][0]
         # else:
         #     false_positives = temp_results['positive'][1] - temp_results['positive'][0]
         # results_list.append(1.0*vals_list[0] / (vals_list[0] + false_positives))
         # # print 'Precision for ' + key + ': ', results_list[1]

         # results_list.append(2*results_list[0]*results_list[1] / (results_list[0] + results_list[1]))
         # # print 'F1 measure for ' + key + ': ', results_list[2]

         # results[key] = results_list

     # return results

def class_test(path, correct_klass, dict_of_catalogs, files_to_test=[]):
     """ Perform classification on a list of files, assumed to be of the same target class. """
     path += '/'+correct_klass
     correct = 0
     total = 0
     if not files_to_test:
         files_to_test = os.listdir(path)
     for name in files_to_test:
     #    print name
        review = loadFile(path + '/' + name)
        sentiment = classify(review, dict_of_catalogs)
        print 'Name: ', name
        print 'Correct classification: ', correct_klass
        print 'Generated Sentiment: ', sentiment
        if sentiment == correct_klass:
            correct += 1
        total += 1

     return [correct, total]

##################################### Provided code

def loadFile(sFilename):
     """Given a file name, return the contents of the file as a string."""
     f = open(sFilename, "r")
     sTxt = f.read()
     f.close()
     return sTxt

def save(data, fileName):
	pickleFile = open(fileName, 'w')
	pickle.dump(data, pickleFile)
	pickleFile.close()

def load(fileName):
	pickleFile = open(fileName, 'r')
	data = pickle.load(pickleFile)
	return data

def main():
    catalog_path = 'catalogs/'
    # teen_catalog = unpickleFile(catalog_path + 'numeric_teen.p')
    # horror_catalog = unpickleFile(catalog_path + 'numeric_horror.p')
    #
    # teen_catalog_frac = generate_percentile_catalog(teen_catalog)
    # horror_catalog_frac = generate_percentile_catalog(horror_catalog)
    #
    # for smooth_factor in [1, .5, .25, .1]:
    #     teen, horror = smooth(load(catalog_path + 'Teen.p'), load(catalog_path + 'Horror.p'), smooth_factor)
    #     catalog_dict = {'Teen': teen, 'Horror': horror}
    #     print 'Smooth factor: ' + str(smooth_factor)
    #     print bulk_test('books', catalog_dict)
    #
    # # test_file_path = 'books/horror/51950.txt'
    # # with open(test_file_path, 'r') as myfile:
    # #     test_text = myfile.read()
    #
    bulk_train('books')

   # books_path = 'books'
   # for folder in os.listdir(books_path):
   #    print '\n\n', folder, '\n\n'
   #    generate_numeric_catalog(books_path + '/' + folder, 'numeric_' + folder.lower() + '.p')

main()
