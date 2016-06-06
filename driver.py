from bayes import *
import copy, sys
########################### Driver functions

def drive_cross_validate():
    # genres = ["Humor", "Adventure", "Science_fiction", "Fantasy", "Young_Adult"]
    folds = 4
    books_path = 'single_genre/'
    genres = [f for f in os.listdir(books_path) if f != '.DS_Store']
    # smoothing_factors = [.00005, .00003, .00001, .000005, .000003, .000001]
    # smoothing_factors = [1.0, 0.5, 0.25, 0.1, 0.01, 0.001, 0.0001]
    sf = sys.argv[1]

    genre_string = ""
    for genre in genres:
        genre_string += genre + '_'
    f = open('results/' + genre_string + str(folds) + "_fold_" + str(sf) + "_smooth", 'w')

    results, accuracies = cross_validate(genres, folds, books_path, float(sf))

    f.write("Averages for smoothing factor of " + str(sf) + ": " + str(results) + '\n')

    # for sf in smoothing_factors:
    #     results, accuracies = cross_validate(genres, folds, books_path, sf)
    #     f.write("Averages for smoothing factor of " + str(sf) + ": " + str(results) + '\n')

    f.close()

def drive_cross_validate_multiple():
    folds = 4
    books_path = 'books_opened/'
    smoothing_factor = 0.1
    genres = ['Adventure', 'Fantasy', 'Historical', 'Horror', 'Humor', 'Literature', 'Mystery', 'New_Adult', 'Other', 'Romance', 'Science_fiction', 'Teen', 'Themes', 'Thriller', 'Vampires', 'Young_Adult']

    macroaverages, metrics = cross_validate_multiple_genres(folds, books_path, smoothing_factor, genres)

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
drive_cross_validate_multiple()
# drive_cross_validate()
# test_smooth_values()
# update_books()
