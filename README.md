# Review-Classification
Review Classification using NPL (Natural Language Processing) "Bag of Words" model.

# Requirements
Python >= 3.0

# Problem description (example)
One restaurant is reviewed by 1000 clients. The Bag of Words model will decide if each review is a negative or a positive one.
The dataset contains two columns: 'Review' and 'Liked', all in a '.tsv' file (columns are separated by TAB). 

# How does it work?
To ensure that the Bag of Words model will return helpful results, the first step should be Preprocessing the data, in such manner that only clean reviews remain.

A clean review includes only relevant words and does NOT include:
    -words such as: 'the', 'a', 'an', 'i' (that are found in the my_stopwords list) 
    -punctuation marks ('...')
    -UPPER-CASE characters

The Bag of Words model creates a sparse matrix with all the words existent in all the reviews on the collums and the actual reviews 
on the rows. For each review, the algorithm counts the number of apparitions of each word, and includes the data in the sparse matrix.
e.g. : If the dataset contains two (clean) reviews ("wow restaurant great", "food awful") the first line of the sparse matrix will be:
```
M=array([[1, 1, 1, 0, 0],[0, 0, 0, 1, 1]])
```
The following classification algorithms are afterwards used:
    -Naive Bayes 
    -Random Forest
    -Logistic Regression
    
# Supported languages (nltk stopwords)
List of supported languages of the nltk.corpus stopwords class.
```
'arabic', 'indonesian', 'swedish', 'romanian', 'nepali', 'greek', 'kazakh', 'italian', 'turkish', 'danish', 'dutch', 'french', 'norwegian', 'english', 'finnish', 'german', 'spanish', 'hungarian', 'portuguese', 'azerbaijani', 'russian'
```
