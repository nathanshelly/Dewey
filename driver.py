from bayes import *
import copy
########################### Driver functions

def drive_cross_validate():
    genres = ["Mystery", "Vampires", "Horror"]
    folds = 4
    books_path = 'books/'
    smoothing_factor = 1

    results, accuracies = cross_validate(genres, folds, books_path, smoothing_factor)

    genre_string = ""
    for genre in genres:
        genre_string += genre + '_'
    f = open('results/' + genre_string + folds + "_fold_" + smoothing_factor + "_smooth", 'w')
    f.write("Averages: " + str(results) + '\n')
    f.write("Per-fold accuracies: " + str(accuracies) + '\n')
    print "Accuracies: ", accuracies
    print "Results: ", results

def test_smooth_values():
    path = 'catalogs/'
    for s_f in [1, .5, .25, .1, .05]:
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

drive_cross_validate()
# test_smooth_values()
# update_books()
