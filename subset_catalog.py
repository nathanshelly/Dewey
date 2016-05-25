import random, copy
from pickling import unpickleFile
from itertools import islice

""" Creating an object to get a book and its features from FB """

def subset(whole, desired = None):
    """ Produces a dictionary of only desired features from a larger features catalog ***for one book """
    if not desired:
        return whole # the whole thing

    return {name:whole[name] for name in desired}

def books_and_features(catalog, des_books = None, des_features = None):
    """ Subsets the entire catalog with only the desired books and features """
    if not des_books and not des_features:
        return catalog # the whole catalog
    if not des_books:
        des_books = catalog.keys() # all the books
    if not des_features:
        return subset(catalog, des_books) # all the features

    return {name:subset(catalog[name], des_features) for name in des_books}

def get_genre_list(genre):
    genres = unpickleFile('genres.p')
    return genres[genre]

def train_and_test(catalog, holdout_per):
    """ Invariant: catalog here has one genre per entry """
    mask = random.sample(range(0, len(catalog)), int(holdout_per*len(catalog)))
    train = {catalog.keys()[i]:catalog.values()[i] for i in range(len(catalog)) if i not in mask}
    test = {catalog.keys()[i]:catalog.values()[i] for i in mask}

    return train, test

def format_catalog_for_classify(catalog):
    new_catalog = []
    for book, features in catalog.iteritems():
        new_catalog.append(({name:value for name, value in features.iteritems() if name != "genre"}, features["genre"])) # only one genre at this point

    return new_catalog

def take(iterable, n):
    return list(islice(iterable, n))

def get_genres(catalog, genres, des_features = None):
    books = []
    for genre in genres:
        books += get_genre_list(genre)
    return books_and_features(catalog, books, des_features)

def only_genres(catalog, genres):
    """ Each book has one genre here """
    return {name:features for name, features in catalog.iteritems() if features["genre"] in genres}
    # return {name:features for name, features in catalog.iteritems() if features["genre"][0] in genres}

def duplicate_genres(catalog):
    new_catalog = {}
    for name, features in catalog.iteritems():
        genres = features["genre"]
        if len(genres) > 1:
            for i in range(len(genres)):
                new_features = copy.deepcopy(features)
                new_features["genre"] = genres[i]
                new_catalog[name+str(i)] = new_features
        else:
            new_features = copy.deepcopy(features)
            new_features["genre"] = genres[0]
            new_catalog[name] = new_features
    return new_catalog


###############################

def reformat_catalog(catalog):
    labels = []
    features_list = []
    for book, features in catalog.iteritems():
        templist = []
        labels.append(features["genre"])
        for feature, value in features.iteritems():
            if feature == "genre":
                continue
            templist.append(value)
        features_list.append(templist)

    return labels, features_list
