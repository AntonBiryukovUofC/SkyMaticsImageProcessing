from sklearn import decomposition
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imfractal import *
from MiscFunctionsSkymatics import *
import cPickle

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC


'''
In this script we try to train the classifier on the bank of labeled images.
Below, the images from three classes given in list_folder are read,
their feature vectors sitting in list_files.

X is then the feature matrix, Y - label matrix.


'''

# Read the DATA part
base = '/home/geoanton/SkyMaticsLearning/DataForClassifiers/'
class_List = ['Bushes','FlatGround','Roads','Damaged','Tracks']
list_folder = [base +('%sEnriched/'% x) for x in class_List]
list_files = ['Feats%s.npz' % x for x in class_List]

for i in range(len(class_List)):
    npzfeats = np.load(list_folder[i]+list_files[i])
    HistsMatrix = npzfeats['HistsMatrix'].astype('int')
    GLCMMatrix = npzfeats['GLCMMatrix']
    FDMatrix = npzfeats['FDMatrix']
    GaborFeats =  npzfeats['GaborFeats']
   # X = np.hstack((HistsMatrix,GLCMMatrix[:,:].squeeze(),GaborFeats[:,:].squeeze(),
    #               FDMatrix[:,:].squeeze()))
    X= np.hstack((HistsMatrix,GLCMMatrix[:,:].squeeze()))
    Y = np.ones(X.shape[0])*i
    if (i==0):
        X_all=X
        Y_all=Y.T
    else:
    
        X_all = np.vstack((X_all,X))
        Y_all= np.hstack((Y_all,Y.T))
    


# Train the classifier part:

    
#X = np.hstack((HistsMatrix,GLCMMatrix[inds,:].squeeze(),FDMatrix[inds,:].squeeze()))
#HistsMatrix = StandardScaler().fit_transform(HistsMatrix)
X= StandardScaler().fit_transform(X_all)
Y=Y_all



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)


# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [100,10,1,1e-2,1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['accuracy']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                       scoring='%s' % score,verbose=1)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()







Classifier = clf
#Classifier.fit(X_train,y_train)
#y_pred = Classifier.predict(X_test)

ConfMatrix  = metrics.confusion_matrix(y_test,y_pred)
ScoreMetric =  metrics.accuracy_score(y_test, y_pred)
print ScoreMetric,"\n",ConfMatrix

fig,ax = plt.subplots()
pca = decomposition.PCA(n_components=2)
XX= pca.fit_transform(X)

ax.scatter(XX[:,0],XX[:,1],c=Y_all,cmap='jet',vmin=0,vmax=len(class_List))
#ax.set_xlim([-15,10])
#ax.set_ylim([-10,10])




# Compute confusion matrix

# Plot non-normalized confusion matrix
class_names = class_List
# Plot normalized confusion matrix
ff =plt.figure()
plot_confusion_matrix(ConfMatrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
with open('ClassifierCanola.pkl', 'wb') as fid:
    cPickle.dump([Classifier, class_names],fid)   


