import argparse
parser = argparse.ArgumentParser(description='Perform k-Means clustering on multilingual string data')
parser.add_argument("path")
args = parser.parse_args()
#Text pre-processing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

nltk.download('stopwords')
nltk.download('wordnet')

#Cleaning the text
import string
def text_process(text):
    '''
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove digits
    3. Remove all stopwords from included languages
   	4. Return the cleaned text as a list of words
    '''
    stemmer = WordNetLemmatizer()
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join([i for i in nopunc if not i.isdigit()])
    nopunc =  [word.lower() for word in nopunc.split() if word not in stopwords.words('english') and word not in stopwords.words('french') and word not in stopwords.words('german') 			and word not in stopwords.words('spanish') and word not in stopwords.words('italian') and word not in stopwords.words('dutch')]
    return [stemmer.lemmatize(word) for word in nopunc]

#Vectorisation : -

from sklearn.feature_extraction.text import TfidfVectorizer

X_train = open(args.path).readlines()
print("File opened")
tfidfconvert = TfidfVectorizer(analyzer=text_process,ngram_range=(1,3)).fit(X_train)
print("File converted")
X_transformed=tfidfconvert.transform(X_train)
print("File transformed")
# Clustering strings using KMean mini batches to save time, as this was designed for a 6 million string set.

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
#Optimal clusters found for this set, using elbow and sil methods.
k=400
out = open("./clust" + str(k) + ".txt","w+")
kmeans = MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=100)
kmeans.fit(X_transformed)
print("Clusters found")
x = {}

print("dumping to file")
for i in range(0, len(kmeans.labels_)):
	if kmeans.labels_[i] in x:
		x[kmeans.labels_[i]].append(X_train[i])
	else:
		x[kmeans.labels_[i]] = [X_train[i]]

for i in x:
	out.write(str(i) + "\t" + str(x[i]) + "\n")
out.close()
print("Done")

