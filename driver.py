from bayes import *
import copy, sys
########################### Driver functions

def drive_cross_validate(folds = 4, books_path = 'books_subset/', smoothing_factor = 0.1, genres = []):
    """ Drives the a cross-validation run, and saves output to the results folder. """
    # folds: number of folds to use
    # books_path: path to a directory containing the books to be used for training and testing
    # smoothing_factor: the smoothing factor to be employed
    # genres: a list of the genres to be tested. To use all genres, give an empty list.

    macroaverages, metrics = cross_validate(folds, books_path, smoothing_factor, genres)

    genre_string = ""
    for genre in macroaverages.keys():
        genre_string += genre + '_'
    f = open('results/' + genre_string + str(folds) + "_fold_" + str(smoothing_factor) + "_smooth_multiple", 'w')
    f.write("Macroaverages: " + str(macroaverages) + '\n')
    f.write("Per-fold metrics: " + str(metrics) + '\n')
    print "Macroaverages: ", macroaverages
    print "Per-fold: ", metrics

def drive_classify(filename):
    """ Drives the classification of a single text. """
    book = loadFile(filename)
    genres = ['Adventure', 'Fantasy', 'Historical', 'Horror', 'Humor', 'Literature', 'Mystery', 'New_Adult', 'Other', 'Romance', 'Science_fiction', 'Teen', 'Themes', 'Thriller', 'Vampires', 'Young_Adult']

    catalogs = {genre:load('catalogs/' + genre + '.p') for genre in genres}
    catalogs = smooth(catalogs, word_list(catalogs), 0.1)
    catalogs = {genre:generate_percentile_catalog(catalog) for genre, catalog in catalogs.iteritems()}

    classifications, probs = classify_text(book, catalogs, len(catalogs.keys()))
    norm = math.fsum(probs)
    probs = [p/norm for p in probs]

    for i in range(len(probs)):
        print "Number " + str(i) + " classification: " + classifications[i] + " (normalized log probability of " + str(probs[i]) + ")"

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "cross":
        drive_cross_validate()
    elif mode == "classify":
        if len(sys.argv) != 3:
            print "Sorry - please supply a filename."
        drive_classify(sys.argv[2])
    else:
        print "Sorry - options are 'cross' or 'classify <filename>'."
