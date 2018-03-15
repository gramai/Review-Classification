#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Cleaning the reviews
def clean_reviews(no_reviews,language, words_to_be_excluded, words_to_be_included, dataset):
    import re
    import nltk
    nltk.download('stopwords') #Must only be downloaded once 
    from nltk.corpus import stopwords #Includes a list of words that souldn't appear in the reivews
    
    #Create own list of stopwords
    my_stopwords = stopwords.words(language) 
    #Append the words from to_be_excluded_words to my_stopwords
    for word in words_to_be_excluded:
        if not any ( word in stpwords for stpwords in my_stopwords):
            my_stopwords.append(word)
    
    #Remove words from my_stopwords that are present in to_be_included
    for word in words_to_be_included:
        if any (word in stpwords for stpwords in my_stopwords):
            my_stopwords.remove(word)
            
    #Stemming is the process of extracting the stem of each word e.g. "loved"->"love"
    from nltk.stem.porter import PorterStemmer
     
    corpus =[] #Will contain the clean reviews"
    for i in range (0, no_reviews):
        review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i]) #Removes all other characters besides the letters and puts the review in a string
        review = review.lower() #Transforms all UPPER-CASE characters to lower-case
        review = review.split() #Separates all the words from the review string and puts them in a list
        ps = PorterStemmer() 
        review = [ps.stem(word) for word in review if not word in set (my_stopwords)] #Remove irrelevant words from rewiews and stems all words  
        review = ' '.join(review) 
        corpus.append(review) #Append review string to corpus
    return corpus

#Creating the bag of words model
def bag_of_words(dataset, corpus, maximum_features):
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features = maximum_features) #Object for creating the sparse matrix 
    X = cv.fit_transform(corpus).toarray() #Sparse matrix 
    y = dataset.iloc[:,1].values #Array with the dependent variable for each review (Liked = 1, Disliked = 0)
    return X,y

def score_acc(cm):
    TN= cm[0][0] #True_Negative or results that were predicted to be negative and were negative
    TP = cm[1][1]#True_Positive or results that were predicted to be positive and were positive
    
    FN = cm[0][1] #False_Negative or results that were predicted to be negative but were positive
    FP = cm [1][0]#False_Positive or results that were predicted to be positive but were negative
    accuracy = (TP + TN)/(TP + TN + FP + FN) #Or correct_predictions/total_predictions
    precision = TP / (TP + FP) #Or correct_positive_predictions/total_correct_predictions
    recall = TP / (TP + FN)#Or correct_positive_predictions/(correct_positive_predictions + false_negative_predictions)
    score = 2 * precision * recall/(precision + recall)
    return accuracy,score

#Applying the Naive Bayes classification algorithm
def naive_bayes(X_train, y_train, X_test, y_test):
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred) #Contains results of predictions
    accuracy,score=score_acc(cm)
    print("Accuracy of the Naive Bayes algorithm is ",  accuracy, " and score ", score)
    return accuracy, score    


#Applying the Random Forest classification algorithm
def random_forest(number_of_trees, X_train, y_train, X_test, y_test):
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = number_of_trees, criterion = 'entropy')
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    accuracy,score=score_acc(cm)
    print("Accuracy of the Random Forest algorithm for ", number_of_trees, " is ",  accuracy, " and score ", score)
    return accuracy, score
      
#Applying the Logistic Regression classification algorithm
def logistic_regression(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    accuracy,score=score_acc(cm)
    print("Accuracy of the Logistic Regression algorithm is ",  accuracy, " and score ", score)
    return accuracy, score


    
