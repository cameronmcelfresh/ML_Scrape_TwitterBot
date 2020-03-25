# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 08:59:56 2019

@author: Cameron
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn import metrics
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix
import itertools


import nltk
nltk.download('wordnet')

import pandas as pd


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.ylim([len(classes)-0.5, -0.5])


"""Remove the outlier topics with the fewest number of articles"""

Topics2Remove = ['HISTORIC PHOTOS', 'MOVIES', 'BOOKS', 'RESTAURANTS', 
                 'PALO ALTO ISSUES', 'AROUND TOWN' ]

for aTopic in Topics2Remove:
    dropIndex = df[df['topic'] == aTopic].index
    df.drop(index = dropIndex, axis = 0, inplace=True) #Drop all rows with that given topic
    
""" Reset the indicies"""
df.reset_index(inplace=True)
df.drop(columns = "index", inplace=True)

import re

"""Clean up the articles! - remove any of the unknown characters and special characters"""
for i in range(len(df)):    
    df["article_text"].loc[i] = re.sub(r'\W', ' ', df["article_text"].loc[i]).lower() #Remove all special symbols and make lowercase
    
    df["article_text"].loc[i] = re.sub(" \d+", " ", df["article_text"].loc[i]) #Remove all digits

    df["article_text"].loc[i] = re.sub(r'\s+[a-zA-Z]\s+', ' ', df["article_text"].loc[i]) #Remove single characters
        
    df["article_text"].loc[i] = re.sub(r'\^[a-zA-Z]\s+', ' ', df["article_text"].loc[i]) #Remove single characters from the start
    
    df["article_text"].loc[i] = re.sub(r'\s+', ' ', df["article_text"].loc[i]) #Change any double spaces to single spaces
    
"""Lemmatize the word - remove any trailing s's"""
from nltk.stem import WordNetLemmatizer 

stemmer = WordNetLemmatizer() 

for i in range(len(df)):
    
    clean_article = df["article_text"].loc[i].split()
    
    clean_article = [stemmer.lemmatize(word) for word in clean_article]
    
    clean_article=' '.join(clean_article)
        
    df["article_text"].loc[i]=clean_article
    
    

df_orig=df #Save a clean copy of the dataframe

############
#############Change the sampling so we have equal representation between article classes
############
from sklearn.utils import resample

df_all = pd.DataFrame()

# Downsample the majority class - becase we have an imbalance of data
for topic in set(df["topic"]):
    temp_df = df[df["topic"]==topic].copy()
    sampled_df = resample(temp_df, replace = False, n_samples = 700, random_state=42)
    
    df_all = pd.concat([df_all, sampled_df])
 
df = df_all


"""Change the topic string to labeled integer value"""
y = df[["topic"]].copy()

from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
y["topic_code"] = lb_make.fit_transform(y["topic"]) #change string to label

y_final = y["topic_code"].copy()


"""Extract the article text"""

articles = df[['article_text']].copy()

articles = list(articles["article_text"].values)

#########################################################################
#########################################################################
"""Explore the data using K-Means Clustering to see if 4 clusters can naturally be found"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

tfidf_vectorizer = TfidfVectorizer(max_df=0.7, max_features=300000,
                                 min_df=0.08, stop_words='english',
                                 use_idf=True, ngram_range=(1,2))

tfidf_matrix = tfidf_vectorizer.fit_transform(articles)

terms = tfidf_vectorizer.get_feature_names()


num_clusters = 4

km = KMeans(n_clusters=num_clusters)

km.fit(tfidf_matrix)

clusters = list(km.labels_) #Take the cluster names

order_centroids = km.cluster_centers_.argsort()[:, ::-1]

for i in range(4):
     print("Cluster %d:" % i, end='')
     for ind in order_centroids[i, :10]:
         print(' %s ,' % terms[ind], end='')
     print()




"""Use multidimensional scaling to visualize the clusters in 2D space"""
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity

MDS()

#convert to 2 components as we're plotting points in a x,y plane
#"precomputed" because we provide a distance matrix that was previously computed
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

dist = 1 - cosine_similarity(tfidf_matrix) #To be used for multidimensional scaling/plotting

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]

cluster_names = {0: 'county, city', 
                 1: 'school, student', 
                 2: 'police, man', 
                 3: 'game, team'}

cluster_df = pd.DataFrame(dict(x=xs, y=ys, label=clusters))

fig, ax = plt.subplots(figsize=(17, 9)) # set size

for key in cluster_names.keys():
    plt.scatter(cluster_df[cluster_df["label"]==key]["x"].values,cluster_df[cluster_df["label"]==key]["y"].values, 
                label = cluster_names[key] )
    
    
ax.legend()
plt.title("K-Means Clustering of Article Text")

plt.xticks([])
plt.yticks([])


#########################################################################
#########################################################################

"""Prepare the text data with bag of words approach"""
#adjust max_df and min_df
#max_df =.7 -> ignore words that occur in more than 70% in documents
#min_df =0.3 -> ignore words that occur in less than 30% of documents
count_vect = CountVectorizer(ngram_range=(1,2) , max_df = 0.7, 
                             strip_accents = "ascii", max_features=300000, 
                             stop_words = "english"  )

X = count_vect.fit_transform(articles) #create a vocab list for bag of words

##Move from occurences to frequencies
tfidf_transformer = TfidfTransformer(use_idf=False)

X_tf = tfidf_transformer.fit_transform(X)

#Do the vectorization and frequency transofrmation in one step
#from sklearn.feature_extraction.text import TfidfVectorizer

#tfidf_vectorizer = TfidfVectorizer(max_df=0.7, max_features=400000,
#                                 min_df=0.08, stop_words='english',
#                                 use_idf=True, ngram_range=(1,3))

#X_tf = tfidf_vectorizer.fit_transform(articles)
 
#Split the data into training and testing groups
X_train, X_test, y_train, y_test = train_test_split(X_tf, y_final, test_size=0.20, random_state=4, shuffle=True)


"""Now try the same thing with just pipelines..."""
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


#Multinomial naive bias classifier
mnb_clf = MultinomialNB()
mnb_parameters = {'alpha':[1e-6, 1e-3, 1], 'fit_prior':('True', 'False')}

gs_mnb = GridSearchCV(estimator=mnb_clf,
			param_grid=mnb_parameters,
			scoring='accuracy',
			cv=5)

#Logistic Regression Classifier
lg_clf = LogisticRegression(random_state=4,multi_class='auto')
lg_parameters = {'solver':('saga','lbfgs'), 'max_iter':[200,500]}

gs_lg = GridSearchCV(estimator=lg_clf,
			param_grid=lg_parameters,
			scoring='accuracy',
			cv=5)

#Random forest classifier
rf_clf = RandomForestClassifier(random_state=4)
rf_parameters = {'n_estimators':[100,200,250], 'bootstrap':("true", "False")}

gs_rf = GridSearchCV(estimator=rf_clf,
			param_grid=rf_parameters,
			scoring='accuracy',
			cv=5)

#Linear Support Vector Machine
svc_clf = LinearSVC(random_state=4)
svc_parameters = {'C':[0.001,0.01,1], 'loss':("hinge", "squared_hinge")}

gs_svc = GridSearchCV(estimator=svc_clf,
			param_grid=svc_parameters,
			scoring='accuracy',
			cv=5)

grids = [gs_mnb, gs_lg, gs_rf,gs_svc ]
grids_title = ['MultinomialNB', 'LogistricRegression', 'RandomForest', 'LinearSVC']

bestAcc=0

for model,model_name in zip(grids, grids_title):
    print(model_name + ":")
    model.fit(X_train, y_train)
    
    print("Best parameters are %s" % model.best_params_)
    
    #Calculate and return the training and test accuracy
    print("Best Training Accuracy: %.3f" % model.best_score_)
    
    y_pred = model.predict(X_test)
    
    test_acc = metrics.accuracy_score(y_test, y_pred)
    
    print("Test Accuracy: %.3f \n" % test_acc)
    
    if (test_acc>bestAcc):
        bestAcc=test_acc
        bestModel = model
        bestCLF = model_name
        
print("The best classifier was "+ bestCLF + " with an accuracy of %.3f" % bestAcc)

"""Explore the predictions of the best model through the confusion matrix"""

y_pred = model.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)

plot_confusion_matrix(conf_matrix, list(lb_make.inverse_transform(range(0,4)))) 



