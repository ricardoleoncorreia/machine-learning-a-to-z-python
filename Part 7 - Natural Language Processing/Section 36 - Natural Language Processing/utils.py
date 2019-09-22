# -*- coding: utf-8 -*-
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def clean_text(column_to_process):
    corpus = []
    for i in range(0, 1000):
        # Remove everything but letter between A and Z
        review = re.sub('[^a-zA-Z]', ' ', column_to_process[i])
        # Set all character to lower case
        review = review.lower()
        # Split to iterate over the review words
        review = review.split()
        # Create an instance of an object to do stemming
        ps = PorterStemmer()
        # Stem words that doesn´t belong to the stopwords list
        # This stopwords list is converted to a set to make the algorithm more efficient
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        # Join the list into a string
        review = ' '.join(review)
        # Add review to corpus list (cleaned dataset)
        corpus.append(review)
    return corpus