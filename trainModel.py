import sys
import numpy
import pandas
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

dataset =  pandas.read_csv("data.csv")

cmts = dataset.loc[:, ['Comment']]
lbls = dataset.loc[:, ['Insult']]

freqvctr = TfidfVectorizer(analyzer='word', stop_words='english')
X = freqvctr.fit_transform(cmts['Comment'])
Y = lbls['Insult']

mnb = MultinomialNB()
mnb = mnb.fit(X,Y)

sgd = SGDClassifier(loss='log', n_iter=100, penalty='elasticnet')
sgd = sgd.fit(X,Y)

rf = RandomForestClassifier(n_estimators=100)
rf = rf.fit(X,Y)

joblib.dump(mnb, 'mnb.model')

joblib.dump(rf, 'rf.model')

joblib.dump(sgd, 'sgd.model')

filename = 'CountVect.cv'
joblib.dump(freqvctr, filename)

ensemble = VotingClassifier(estimators=[('nb', mnb), ('sgd', sgd), ('rf', rf)])
ensemble = ensemble.fit(X,Y)

joblib.dump(ensemble, 'ensemble.model')
