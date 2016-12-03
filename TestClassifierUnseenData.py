import PIL
import numpy as np
import glob
import cPickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from shutil import copyfile
from skimage.feature import greycomatrix, greycoprops
from MiscFunctionsSkymatics import *
import os
    
parent_dir = './images_all_NoFrame/'
list_im = glob.glob(parent_dir + '*.png')

n_images = 150
ind = np.random.choice(range(0,len(list_im)),n_images)
list_im = [list_im[x] for x in ind]


pixs = [2,4,8,16]
angles = [90,180,270,360]
frqs = (0.1,0.5,1)
kernels = GetKernels(frqs)
reread_Hists=True
if reread_Hists:
    with open('kmeansadlee.pkl', 'rb') as fid:
        k_means = cPickle.load(fid)
    HistsMatrix = np.zeros((len(list_im),k_means.n_clusters))
    # Two features from the GLCM
    GLCMMatrix =  np.zeros((len(list_im),2*len(pixs)*len(angles)))
    GaborFeats = np.zeros((len(list_im),len(frqs)*2*3))

    i=0
    for image_file in list_im:
        image = PIL.Image.open(image_file)
        arrayIm,greyImage=predictImageFromKmeans(k_means,image)
        ImagesMatrix = arrayIm
        HistsMatrix[i,:] = np.histogram(ImagesMatrix,bins=range(k_means.n_clusters+1))[0]
        GLCM  = greycomatrix(greyImage.astype('uint8'), pixs, angles, symmetric=True, normed=True)
        GaborFeats[i,:] = GetFeaturesGaborFilters(greyImage,kernels).flatten()

        GLCMMatrix[i,:] = np.hstack((greycoprops(GLCM, 'dissimilarity').flatten(),greycoprops(GLCM, 'correlation').flatten()))
        i+=1
        print 'Doing image number %d out of %d' %(i+1,len(list_im))
      
size = 65536.0
const = 0.12
# Reject black images 
inds = np.where(HistsMatrix[:,1]/size < const)[0]
HistsMatrix = HistsMatrix[inds,:].squeeze()
list_im = np.array(list_im)[inds]
X = np.hstack((HistsMatrix,GLCMMatrix[:,:].squeeze(),GaborFeats[:,:].squeeze()))
X= StandardScaler().fit_transform(X)
with open('./ClassifierCanola.pkl', 'rb') as fid:
    [Classifier, class_names] = cPickle.load(fid)
y_pred = Classifier.predict(X)

new_folder = './Classified/'
if not(os.path.exists(new_folder)):
    os.makedirs(new_folder)
    
for i,pred_item in enumerate(y_pred):
    fname = list_im[i]
    base_name = fname.split('/')[-1]

    new_name = new_folder+"%s_%s" %(class_names[int(pred_item)],base_name)
    copyfile(fname,new_name)
    print ' Saved / classified %d out of %d' %(i,len(list_im))
    
from sklearn import decomposition

fig,ax = plt.subplots()
pca = decomposition.PCA(n_components=2)
XX= pca.fit_transform(X)

ax.scatter(XX[:,0],XX[:,1],c=y_pred,cmap='jet')
