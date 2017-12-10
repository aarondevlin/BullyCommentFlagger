import numpy as np
import pandas
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from scipy import interp
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import RidgeClassifier


np.random.seed(123)

# Load training data set
data =  pandas.read_csv("data.csv")

# Fit and transform data to TfidifVectorizer
freqvctr = TfidfVectorizer(analyzer='word', stop_words='english')
X = freqvctr.fit_transform(data['Comment'])
Y = data['Insult']

# Split the data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)

# Create Classifiers
mnb = MultinomialNB()
sgd = SGDClassifier(loss='log', n_iter=100, penalty='elasticnet')
rf = RandomForestClassifier(n_estimators=100)
ensemble = VotingClassifier(estimators=[('nb', mnb), ('sgd', sgd), ('rf', rf)])

classifiers = [mnb, sgd, rf, ensemble]
classifierNames = ['Multinomial Naive Bayes', 'SGD', 'Random Forest', 'Ensemble']

# False positive rates, true positive rates, and roc dictionaries
fpr = {}
tpr = {}
rocAuc = {}

# Fit training data, predict testing data, print evaluations,
# and save corresponding fpr, tpr, and rocAuc
for i in range(len(classifiers)):
    print("{} Classifier Score:".format(classifierNames[i]))
    classifiers[i].fit(Xtrain, Ytrain)
    preds = classifiers[i].predict(Xtest)

    kf = KFold(n_splits = 10, random_state = 0)
    scores = cross_val_score(classifiers[i], X, Y, cv=kf)
    print("Accuracy: %.4f" % scores.mean())

    tn, fp, fn, tp = confusion_matrix(Ytest, preds.round()).ravel()

    print("Precision: %.4f" % (tp/(tp+fp)))
    print("False Positive Rate: %.4f" % (fp/(fp + tn)))
    print("Recall: %.4f" % (tp/(tp + fn)))
    print("\n")

    fpr[i], tpr[i], _ = roc_curve(Ytest, preds)
    rocAuc[i] = auc(fpr[i], tpr[i])

# Array of all false positive rates
fprAll = np.unique(np.concatenate([fpr[i] for i in range(len(classifiers))]))

# Interpolate all roc curves
tprMean = np.zeros_like(fprAll)
for i in range(len(classifiers)):
    tprMean += interp(fprAll, fpr[i], tpr[i])

# Average and compute AUC
tprMean /= 4

# Plot roc curves
plt.figure()

colors = cycle(['cornflowerblue', 'darkorange', 'deeppink', 'darkgreen'])

for i, color in zip(range(4), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='{0} (area = {1:0.2f})'.format(classifierNames[i], rocAuc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Bully Flagger ROC Curves')
plt.legend(loc="lower right")
roc = plt.gcf()
plt.show()
plt.draw()
roc.savefig('roc_curve.png', dpi=100)
