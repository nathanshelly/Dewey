from collections import Counter
import os
from nltk.corpus import stopwords

c = Counter()

for i in os.listdir(os.getcwd()):
	if i.endswith(".txt"):
		file = open(str(i))
		for line in file:
			for word in line.split():
				c[word.lower().strip(".',")] += 1

most_common_500 = c.most_common(500)
stop_words = stopwords.words('english')
stop_words = [word.lower() for word in stop_words]

most_common_not_stopwords = []

for word, count in most_common_500:
	if not word in stop_words and (str(word)[-2:0] != 'ci'):
		most_common_not_stopwords.append(word)

print most_common_not_stopwords[:100]