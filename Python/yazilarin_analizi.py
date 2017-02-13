import nltk
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

dataset = pd.ExcelFile('SourceFile.xlsx')
dataset = dataset.parse('Titled',header=None)

postTxt = dataset.iloc[:,0]
postDocs = [x for x in postTxt]
postDocs = [x.lower() for x in postDocs]

stopset = set(stopwords.words('english'))

# Önceden deneme icin
postDocs[0]

vectorizer = TfidfVectorizer(stop_words = stopset, use_idf = True, ngram_range = (1, 2))
X = vectorizer.fit_transform(postDocs)
#print (X[0])
X.shape

lsa = TruncatedSVD(n_components=20, n_iter = 100)
lsa.fit(X)

#Get the document matrix
dtm_lsa = lsa.fit_transform(X)
dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)

a = lsa.components_
terms = vectorizer.get_feature_names()
for i, comp in enumerate(lsa.components_):
    termsInComp = zip (terms, comp)
    sortedTerms = sorted(termsInComp, key = lambda x: x[1], reverse = True)[:10]
    print ('Word Group %d:' % i)
    for term in sortedTerms:
        print (term[0], term[1])
    print(" ")

 # Compute document similarity using LSA components
similarity = np.asarray(np.asmatrix(dtm_lsa) * np.asmatrix(dtm_lsa).T)
b = pd.DataFrame(similarity,index=X, columns=X).head(10)