from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from gensim import corpora, models
import gensim
import pandas as pd
import cPickle as pickle

# create English stop words list
eng_stopwords = set(stopwords.words('english'))

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

#load documents
df = pd.read_table('plot_summaries.txt', sep='\t', header=None)
moviePlotSeries = df[1].str.decode('utf-8', "ignore")
moviePlotList = moviePlotSeries.tolist()

# list for tokenized documents in loop
texts = []

# loop through document list
for i in moviePlotList:

    # clean and tokenize document string
    raw = i.lower()
    tokens = word_tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in eng_stopwords]

    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

    # add tokens to list
    texts.append(stemmed_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
ldamodel = gensim.models.ldamulticore.LdaMulticore(corpus, num_topics=100, id2word = dictionary, passes=20, workers=39)

with open('movieModel.pkl', 'w') as f:
        pickle.dump(ldamodel, f)
