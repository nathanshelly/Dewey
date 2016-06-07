from bayes import *
import copy, sys
########################### Driver functions

def drive_cross_validate(folds = 4, books_path = 'books_opened/', smoothing_factor = 0.1, genres = ['Fantasy', 'Horror']):
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

if __name__ == "__main__":
    drive_cross_validate()
