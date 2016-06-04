from bayes import *
import copy
########################### Driver functions

def drive_cross_validate():
    # genres = ["Humor", "Adventure", "Science_fiction", "Fantasy", "Young_Adult"]
    folds = 4
    books_path = 'single_genre/'
    genres = [f for f in os.listdir(books_path) if f != '.DS_Store']
    smoothing_factors = [.00005, .00003, .00001, .000005, .000003, .000001]

    genre_string = ""
    for genre in genres:
        genre_string += genre + '_'
    f = open('results/' + genre_string + str(folds) + "_fold_multiplesmoothing", 'w')

    for sf in smoothing_factors:
        results, accuracies = cross_validate(genres, folds, books_path, sf)
        f.write("Averages for smoothing factor of " + str(sf) + ": " + str(results) + '\n')

    f.close()

def drive_cross_validate_multiple():
    folds = 4
    books_path = 'books_opened/'
    smoothing_factor = 0.0005

    macroaverages, metrics = cross_validate_multiple(folds, books_path, smoothing_factor)

    genre_string = ""
    for genre in macroaverages.keys():
        genre_string += genre + '_'
    f = open('results/' + genre_string + str(folds) + "_fold_" + str(smoothing_factor) + "_smooth_multiple", 'w')
    f.write("Macroaverages: " + str(macroaverages) + '\n')
    f.write("Per-fold metrics: " + str(metrics) + '\n')
    print "Macroaverages: ", macroaverages
    print "Per-fold: ", metrics

def test_smooth_values():
    path = 'catalogs/'
    f = open('smoothing_results.txt', 'w')
    f.write('')
    f.close()
    for s_f in [.00005, .00003, .00001, .000005, .000003, .000001]: # [1, .5, .25, .1, .05, .04, .03, .02, .01, .005, .003, .001, .0005, .0003, .0001]:
        f = open('smoothing_results.txt', 'a')
        f.write('Smoothing factor: ' + str(s_f) + '\n')
        f.close()
        print 'Smoothing factor:', s_f
        catalogs = {'Teen': load(path + 'Teen.p'), 'Horror': load(path + 'Horror.p')}
        words = word_list(catalogs)
        catalogs = smooth(catalogs, words, s_f)
        for key in catalogs.keys():
            catalogs[key] = generate_percentile_catalog(catalogs[key])
        bulk_test(catalogs)

def update_books():
    books = load('books.p')
    temp = {}
    for key in books.keys():
        temp[key] = books[key]['genre']
    save(temp, 'books_genres.p')
    return temp

# split_genres()
# drive_cross_validate_multiple()
drive_cross_validate()
# test_smooth_values()
# update_books()
