import cPickle as pickle

def loadFile(sFilename):
    '''Given a file name, return the contents of the file as a string'''
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
