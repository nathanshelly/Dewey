from bayes import *

########################### Driver functions

def set_aside_data():
    for value in variable:
        pass

def drive_cross_validate():
    genres = ["Mystery", "Vampires"]
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
    catalog_path = 'catalogs/'
    smoothed_path = 'catalogs_smoothed/'
    master_word_list = loadFile('all_words_list.p')
    smoothed_ending = '_smoothed.p'
    for s_f in [1, .5, .25]:
        print "Smoothing value", s_f
        smooth_all(catalog_path, smoothed_path, master_word_list, s_f)
        catalogs = {'Teen': generate_percentile_catalog(load(smoothed_path + 'Teen' + smoothed_ending)), 'Horror': generate_percentile_catalog(load(smoothed_path + 'Horror' + smoothed_ending))}
        bulk_test(catalogs)

drive_cross_validate()
