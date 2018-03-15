#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def demo():
    #Please update values of the following variables:
    path = 'Restaurant_Reviews.tsv'
    language = 'english' #You can find the list of accepted languages in the README.md file
    words_to_be_included = ['not','nor','no'] #That should be part the reviews
    words_to_be_excluded = ['opinion'] #That shouldn't be part of the reviews
    no_reviews = 1000
    maximum_features = 1500 #Maximum elements that the Bag of Words could contain
    test_set_size = 0.2 #Percentage of the test set
    
    #Importing the dataset
    import pandas as pd 
    dataset = pd.read_csv(path, delimiter = '\t', quoting = 3) 
    #delimiter = '\t' indicates TAB as delimiter, opposed to the clasical delimiter ","
    #quoting = 3 ignores the "" present in the reviews
    
    corpus = review_classification.clean_reviews(no_reviews,language,words_to_be_excluded,words_to_be_included,dataset)    
    
    X,y = review_classification.bag_of_words(dataset,corpus,maximum_features)

    # Splitting the dataset into the Training set and Test set
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_set_size)
    
    #Naive Bayes
    accuracy_bayes,score_bayes=review_classification.naive_bayes(X_train,y_train,X_test,y_test)

    #Random Forest
    number_of_trees=1000
    accuracy_random_forest,score_random_forest=review_classification.random_forest(number_of_trees, X_train,y_train,X_test,y_test)

    #Logistic Regression
    accuracy_LR,score_LR=review_classification.logistic_regression(X_train,y_train,X_test,y_test)
    
if __name__ == "__main__":
    import review_classification
    demo()