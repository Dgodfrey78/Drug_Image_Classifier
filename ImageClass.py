import os
from IPython.display import display

import matplotlib as mpl
import matplotlib.pyplot as plt

import cv2
import pandas as pd
import numpy as np
from numpy import asarray
from PIL import Image
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from EdgeDetect import edge
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_curve, auc

accurate = 0.0

labels = pd.DataFrame({'Drug':['weed','weed','weed','weed','weed','weed','cocaine','cocaine','cocaine','cocaine','cocaine','heroin','heroin','heroin','heroin','heroin','heroin','heroin','heroin','heroin','heroin','cocaine','cocaine','cocaine','cocaine','cocaine','weed','weed','weed','weed','weed','heroin','heroin','heroin','heroin','heroin','cocaine','cocaine','cocaine','cocaine','cocaine','weed','weed','weed','weed','weed','heroin','heroin','heroin','heroin','heroin','cocaine','cocaine','cocaine','cocaine','cocaine','heroin','heroin','heroin','heroin','heroin','weed','weed','weed','weed','weed','cocaine','cocaine','cocaine','cocaine','cocaine','weed','weed','weed','weed','weed','heroin','heroin','heroin','heroin','heroin','cocaine','cocaine','cocaine','cocaine','cocaine','weed','weed','weed','weed','weed','heroin','heroin','heroin','heroin','heroin','cocaine','cocaine','cocaine','cocaine','cocaine','weed','weed','weed','weed','weed','heroin','heroin','heroin','heroin','heroin','cocaine','cocaine','cocaine','cocaine','cocaine','weed','weed','weed','weed','weed','heroin','heroin','heroin','heroin','heroin','cocaine','cocaine','cocaine','cocaine','cocaine','weed','weed','weed','weed','weed','heroin','heroin','heroin','heroin','heroin','cocaine','cocaine','cocaine','cocaine','cocaine','weed','weed','weed','weed','weed','heroin','heroin','heroin','heroin','heroin','cocaine','cocaine','cocaine','cocaine','cocaine','weed','weed','weed','weed','weed','heroin','heroin','heroin','heroin','heroin','cocaine','cocaine','cocaine','cocaine','cocaine','weed','weed','weed','weed','weed','heroin','heroin','heroin','heroin','heroin','cocaine','cocaine','cocaine','cocaine','cocaine','cocaine','cocaine','cocaine','cocaine','cocaine','cocaine','cocaine','cocaine','cocaine','cocaine','cocaine','cocaine','cocaine','weed','weed','weed','weed','weed','weed','weed','weed','weed','weed','weed','weed','weed','weed','weed','weed','weed','weed','weed','weed','weed','weed','weed','weed','weed']})

labels.head()

def get_image(row_id, root="datasets/"):

    filename = "{}.jpg".format(row_id)
    file_path = os.path.join(root, filename)
    img = Image.open(file_path)
    return np.array(img)
    
def get_image_recog(row_id):

    filename = "{}.jpg".format(row_id)
    
    img = Image.open(filename)

    return np.array(img)
def create_features(img):
    #resize image but keep dimensionality
    sizedimg= np.resize(img,(150,150,3))
    #create matrix in 2D
    RGBaverage = np.zeros((150,150))
    
    #loop through RBG photo pixel by pixel and get the average
    for i in range(0,sizedimg.shape[0]-1):
        for j in range(0,sizedimg.shape[1]-1):
            RGBaverage[i][j] = ((int(sizedimg[i,j,0]) + int(sizedimg[i,j,1]) + int(sizedimg[i,j,2]))/3)
    
    flat_color = RGBaverage.flatten()
    # flatten three channel color image
    color_features = sizedimg.flatten()
    # Use rgb2gray to convert to grascale
    gray_image = rgb2gray(sizedimg)
    # get HOG features from greyscale image
    hog_features = hog(gray_image, block_norm='L2-Hys', pixels_per_cell=(16, 16))
    # combine color and hog features into a single array
    flat_features = np.hstack((hog_features,flat_color,color_features))
    return flat_features
    
def create_feature_matrix(label_dataframe):
    features_list = []

    for img_id in label_dataframe.index:
        img = get_image(img_id)
       
 
 
        image_features = create_features(img)
        features_list.append(image_features)
    
    #make feature matrix
    feature_matrix = np.array(features_list)
    
    
 

    # use standard scaler
    ss = StandardScaler()
    # run this on our feature matrix
    stand_feat = ss.fit_transform(feature_matrix)

    pca = PCA(n_components=200)
    # use fit_transform to run PCA on our standardized matrix
    pca_feat = ss.fit_transform(stand_feat)
    # look at new shape
    
    
    X = pd.DataFrame(pca_feat)
    y = pd.Series(labels.Drug.values)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=.2,
                                                        random_state=1234123)

    
    kernel = 1.0 * RBF(1.0)
    svm = SVC(kernel='poly',probability=True, random_state=42, gamma=3)
    clf = DecisionTreeClassifier(random_state=0)
    cross_val_score(clf,X_test,y_test, cv=10)

    # fit model
    svm.fit(X_train, y_train)
    clf.fit(X_train, y_train)
    y_preds = clf.predict(X_test)
    print(clf.score(X_test,y_test))
    
    y_pred = svm.predict(X_test)
    
   
   
    
    print(X_test)
    #imgd =get_image('ima')

    feature_l= []

    # calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    global accurate
    accurate = accuracy
    print('Model prediction is: ',y_pred)
    print('Model accuracy is: ', accuracy)
    
    return svm
    
svm = create_feature_matrix(labels)
def whichDrug(filepath):
    global accurate
    

    imgd =get_image_recog(filepath)
    create=create_features(imgd)
    create1=create.reshape(1,-1)
    y_preds = svm.predict(create1)
    
    print(y_preds)
    titled = ("{} with {}% confidence").format(y_preds,accurate)
    plt.imshow(get_image_recog(filepath))
    
    plt.title(titled)
    plt.show()
