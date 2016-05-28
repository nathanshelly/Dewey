from bayes import *

########################### Driver functions

def drive_cross_validate():
    genres = ["Teen", "Horror"]
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
    catalogs = {'Teen': load(path + 'Teen.p'), 'Horror': load(path + 'Horror.p')}
    words = word_list(catalogs)
    for s_f in [1, .5, .25]:
        catalogs = smooth(catalogs, words, s_f)
        for key in catalogs.keys():
            catalogs[key] = generate_percentile_catalog(catalogs[key])
    bulk_test(catalogs, divisor = 10)

# drive_cross_validate()
test_smooth_values()
