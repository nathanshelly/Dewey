import re, string, nltk.data, nltk

TP_NAME = 'tp_freq'
FP_NAME = 'fp_freq'
AV_LEN_NAME = 'mean_len'
HAS_COCK = 'cock_freq'
TOP_99 = '99_words_freq' # as currently written, must be the last feature in list (later features will not be calculated)
# HAS_VAMPIRE = 'vamp_freq'
UNIQUE = 'unique_words'
LENGTH = 'length'
UNIQUE_FREQ = 'unique_freq'

top_words = ['said', 'back', 'like', 'would', 'one', 'could', 'know', 'i', 'eyes', 'time', 'get', "didn't", 'around', 'even', 'going', 'head', 'way', 'see', 'want', 'looked', "don't", 'hand', 'think', 'away', 'still', "i'm", 'face', 'go', 'never', 'right', 'asked', 'thought', 'something', 'made', 'knew', 'much', 'two', 'us', 'look', 'little', 'man', 'door', 'got', 'make', 'take', '"you', 'room', 'felt', 'good', 'wanted', "he'd", 'took', 'long', "wasn't", 'turned', 'first', "it's", 'need', 'come', 'say', 'hands', 'let', "she'd", 'voice', 'tell', 'really', 'told', 'came', 'left', 'another', 'sure', 'last', 'body', 'people', 'feel', 'went', 'life', 'behind', 'well', "couldn't", 'anything', 'side', 'enough', 'saw', 'might', "i'd", "you're", 'put', 'every', 'night', 'nothing', 'find', 'day', 'thing', 'ever', 'things', 'though', 'love', 'hair']

def main():
	''' Function to test these classes'''
	fe = feature_extractor('books/Adventure/219.txt')
	freq_list = fe.extract(['99_words_freq'])
	print freq_list

class feature_extractor:
	'''A class to extract features from a specified path'''
	def __init__(self, path, existing_features):
		self.filepath = path
		self.file = open(path)
		self.existing_features = existing_features

	def word_extract(self, list_of_feature_names):
		'''Given a list of features to be extracted, returns a dictionary of the feature names keyed to their values'''
		counters = self.gen_counters(list_of_feature_names)

		if len(counters) > 1:
			for line in self.file:
				for word in line.split():
					for counter in counters:
						counter.update(word.translate(None, string.punctuation).lower())
		total_word_count = counters[0].output()
		counters = counters[1:]
		features = {}
		for i in range(len(list_of_feature_names)):
			feature_name = list_of_feature_names[i]

			if feature_name == TOP_99:
				# this bit is super hacky
				for word in top_words:
					features[word+'_freq'] = counters[i].output(total_word_count)
					i += 1
				break
			else:
				features[list_of_feature_names[i]] = counters[i].output(total_word_count)
			# value = counter.count #1.0 * counter.count / total_word_count
			# feature_values.append(counter.output(total_word_count))
		return features

	def file_extract(self, list_of_feature_names):
		features = {}

		for feat in list_of_feature_names:
			if feat == UNIQUE:
				features[UNIQUE] = len(self.unique_words())
			if feat == LENGTH:
				features[LENGTH] = len(self.all_words())
		return features

	def existing_extract(self, list_of_feature_names):
		features = {}

		for feat in list_of_feature_names:
			if feat == UNIQUE_FREQ:
				features[UNIQUE_FREQ] = 1.0*self.existing_features[UNIQUE]/self.existing_features[LENGTH]
		return features


	def sentence_extractor(self, sentence_length = False):
		pass
		# sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
		# tokenize(self.file.read().strip())

	def all_words(self):
		f = self.file.read()
		f = f.decode('ascii', 'ignore')
		f = f.encode('ascii')
		self.file = open(self.filepath)

		return nltk.word_tokenize(f)

	def unique_words(self):
		return set(self.all_words())


	def gen_counters(self, list_of_feature_names):
		'''Generates and returns the appropriate counters given a list of features'''
		counters = []
		counters.append(all_word_counter())
		for feature_name in list_of_feature_names:
			if feature_name == TP_NAME:
				counters.append(list_word_counter(['he', 'she','his','hers','him','her']))
				continue
			elif feature_name == FP_NAME:
				counters.append(list_word_counter(['i', 'me','my','mine']))
				continue
			elif feature_name == AV_LEN_NAME:
				counters.append(word_length_counter())
			elif feature_name == HAS_COCK:
				counters.append(list_word_counter(['cock', 'dick', 'schlong', 'weiner', 'shaft', 'balls', 'bratwurst', 'johnson', 'dongle', 'penis']))
			elif feature_name == TOP_99:
				for word in top_words:
					counters.append(list_word_counter([word]))
			elif feature_name == 'vamp_freq':
				counters.append(list_word_counter(['vampire', 'vampires', 'vamps', 'dracula']))
		return counters

	def word_iter(self, book, funcs):
		'''Unused function to apply a function on each word in the book'''
		for line in book:
			for word in line.split():
				for func, args in funcs:
					func(*args)

class all_word_counter:
	'''Counter to count all words in a book'''
	def __init__(self):
		self.count = 0
	def update(self,word):
		'''Counts any word'''
		self.count += 1
	def output(self):
		return self.count

class list_word_counter:
	'''Counter to count all words in a given list'''
	def __init__(self,word_list):
		self.count = 0
		self.words = word_list
	def update(self,word):
		'''counts word if in list'''
		if word in self.words:
			self.count += 1
	def output(self,total_word_count):
		'''Return frequency of word'''
		return 1.0*self.count / total_word_count

class word_length_counter:
	'''Counter to count mean length of all words'''
	def __init__(self):
		self.total_letters = 0
	def update(self,word):
		'''Adds number of letters to running total'''
		self.total_letters += len(word)
	def output(self, total_word_count):
		'''Divides total number of letters by number of words to get mean length'''
		return 1.0*self.total_letters / total_word_count
