import numpy as np
import pandas as pd
# from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn import manifold
#importing nlp library
import nltk
#library that contains punctuation
import string
#importing the Stemming function from nltk library
from nltk.stem.porter import PorterStemmer
#defining function for tokenization
import re

#defining the function to remove punctuation
def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

def tokenization(text):
    tokens = re.split(r' ',text)
    return tokens

#Stop words present in the library
stopwords = nltk.corpus.stopwords.words('english')

#defining the function to remove stopwords from tokenized text
def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output

#defining the object for stemming
porter_stemmer = PorterStemmer()

#defining a function for stemming
def stemming(text):
    stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text

# from nltk.stem import WordNetLemmatizer
# #defining the object for Lemmatization
# wordnet_lemmatizer = WordNetLemmatizer()
# #defining the function for lemmatization
# def lemmatizer(text):
#     lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
#     return lemm_text
# data['msg_lemmatized']=data['no_stopwords'].apply(lambda x:lemmatizer(x))

# def display_closestwords_tsnescatterplot(model, word, size):

    arr = np.empty((0,size), dtype='f')
    word_labels = [word]
    close_words = model.wv.similar_by_word(word)
    arr = np.append(arr, np.array([model.wv[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model.wv[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
            
    tsne = manifold.TSNE(n_components=2, pca=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    plt.scatter(x_coords, y_coords)
    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
        plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
        plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()

def data_preprocessing(path):
    imdb_data=pd.read_csv(path, encoding="unicode_escape")
    imdb_data['sentiment'] = imdb_data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

    #storing the puntuation free text
    imdb_data['review']= imdb_data['review'].apply(lambda x:remove_punctuation(x))
    imdb_data['review']= imdb_data['review'].apply(lambda x: x.lower())
    
    #applying function to the column
    imdb_data['review']= imdb_data['review'].apply(lambda x: tokenization(x))
    
    #applying the function
    imdb_data['review']= imdb_data['review'].apply(lambda x:remove_stopwords(x))
    imdb_data['review'] = imdb_data['review'].apply(lambda x: stemming(x))

    print('X.shape:', imdb_data.iloc[:,:-1].shape)
    print('y.shape:', imdb_data.iloc[:,-1].shape)

    return imdb_data


if __name__ == '__main__':
    imdb_data = data_preprocessing(r'D:\JunShen\nlpPractice\IMDB Dataset.csv')