### calculates top 10 most similar book plots to Will Wonka and the Chocolate Factory plot

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from gensim import corpora, models
import gensim
import pandas as pd
import cPickle as pickle

with open('bookModel.pkl') as f:
    bookModel = pickle.load(f)

with open('movieModel.pkl') as f:
    movieModel = pickle.load(f)

movie_df = pd.read_table('plot_summaries.txt', sep='\t', header=None)

###Willy Wonka and the Chocolate Factory
movie = movie_df[1].loc[movie_df[0]==174560]

moviePlot = movie.iloc[0].decode('utf-8', "ignore")

# create English stop words list
eng_stopwords = set(stopwords.words('english'))

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

tokenized_plot = word_tokenize(moviePlot.lower())
stopped_plot = [i for i in tokenized_plot if not i in eng_stopwords]
stemmed_plot = [p_stemmer.stem(i) for i in stopped_plot]

df = pd.read_table('book_plots.txt', sep='\t', header=None)
bookPlotSeries = df[6].str.decode('utf-8', "ignore")
bookPlotList = bookPlotSeries.tolist()

texts = []

# loop through document list
for i in bookPlotList:

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
book_corpus = [dictionary.doc2bow(text) for text in texts]


### build and save book plot index to calculate similarity with movie plot
###index = gensim.similarities.MatrixSimilarity(bookModel)
###index.save("bookModel.index")

index = gensim.similarities.MatrixSimilarity.load('bookModel.index')

movie_bow = dictionary.doc2bow(stemmed_plot) ##doc2bow for Willy Wonka plot
movie_lda = movieModel[movie_bow] ## topic distribution in Willy Wonka plot

##similarity in topic distribution between Willy Wonka plot and book plots in index
sims = index[movie_lda] 
sims = sorted(enumerate(sims), key=lambda item: -item[1])

##select 10 book plots with most similar distribution

top10 = sims[0:10]

print top10



