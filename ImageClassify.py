import os
from IPython.display import display

import matplotlib as mpl
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np
from numpy import asarray
from PIL import Image

from skimage.feature import hog
from skimage.color import rgb2gray

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn.metrics import roc_curve, auc

accurate=0.0
labels = pd.DataFrame({'Drug':['drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','not drugs','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug','drug']})

labels.head()



def get_image(row_id, root="dataset/"):

    filename = "{}.jpg".format(row_id)
    file_path = os.path.join(root, filename)
    img = Image.open(file_path)
    return np.array(img)
    
def get_image_recog(row_id):

    filename = "{}.jpg".format(row_id)

    img = Image.open(filename)

    return np.array(img)
    
def create_features(img):
    sizedimg= np.resize(img,(150,150))
    # flatten color image
    color_features = sizedimg.flatten()
    # use rgb2gray to convert to grayscale
    gray_image = rgb2gray(sizedimg)
    # Hog Fetures derived from grayscale
    hog_features = hog(gray_image, block_norm='L2-Hys', pixels_per_cell=(16, 16))
    # combine color and hog features into a single array
    flat_features = np.hstack((color_features,hog_features))
    return flat_features
    
def create_feature_matrix(label_dataframe):
    features_list = []

    for img_id in label_dataframe.index:
    
       
        img = get_image(img_id)
       
        # get features for image
        image_features = create_features(img)
        features_list.append(image_features)
    
    # convert list of arrays into a matrix
    feature_matrix = np.array(features_list)
    
    
    # get the standard 
    ss = StandardScaler()
    
    feat_stand = ss.fit_transform(feature_matrix)

    pca = PCA(n_components=500)
    # use fit_transform to run PCA on our standardized matrix
    pca_feat = ss.fit_transform(feat_stand)
   
   
    
    X = pd.DataFrame(pca_feat)
    y = pd.Series(labels.Drug.values)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=.2,
                                                        random_state=1234123)

    
    print(pd.Series(y_train).value_counts())
    
    svm = SVC(kernel='poly', probability=True, random_state=42, gamma=0.01)

    # fit model
    svm.fit(X_train, y_train)
    
    y_pred = svm.predict(X_test)
    
   
    print(X_test)

    feature_l= []

    global accurate
    accuracy = accuracy_score(y_test, y_pred)
    accurate=accuracy
    print('Model prediction is: ',y_pred)
    print('Model accuracy is: ', accuracy)
    
    return svm
    
svm = create_feature_matrix(labels)
def isDrug(filepath):
    global accurate
    #print(feature_matrix.shape)

    imgd =get_image_recog(filepath)
    create=create_features(imgd)
    create1=create.reshape(1,-1)
    y_preds = svm.predict(create1)
    print(y_preds)
    titled = ("{} with {}% confidence").format(y_preds,accurate)
    plt.imshow(get_image_recog(filepath))
    plt.title(titled)
    plt.show()
