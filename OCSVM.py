import numpy as np
import keras.backend as K
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam
from keras import callbacks
from keras.initializers import VarianceScaling
from keras.losses import mse
from sklearn.decomposition import PCA
from sklearn import svm
import matplotlib.pyplot as plt
import cv2
import os

model = load_model('ConvAE50Tulelake40_v3.h5')

X_train = []
X_test = []
count = 0

for root, directory, files in os.walk("./swamp_aug_Tulelake"):
    for fname in files:
        if count > 99:
            break
        else:
            X_train.append(cv2.imread(os.path.join(root, fname),).astype('float64'))
            count += 1
            
for root, directory, files in os.walk("./imgTulelakeSample"):
    for fname in files:
        X_test.append(cv2.imread(os.path.join(root, fname),).astype('float64'))
        
# for root, directory, files in os.walk("./cluster1"):
#     for fname in files:
#         X_train.append(cv2.imread(os.path.join(root, fname),).astype('float64'))
            
# for root, directory, files in os.walk("./imgTulelake"):
#     for fname in files:
#         X_test.append(cv2.imread(os.path.join(root, fname)).astype('float64'))


image_size = 50
X_train = np.array(X_train)
X_test = np.array(X_test)
# X_train = np.expand_dims(X_train, axis=-1)

X_train /= 255.0
X_test /= 255.0

# X_train = np.reshape(X_train, [-1, original_dim]) 
print ("X_train shape: ", X_train.shape)
print ("X_test shape: ", X_test.shape)

encoder = Model(inputs=model.input, outputs=model.get_layer('encoder').output) #latent_vector2
# encoder = Model(inputs=model.input, outputs=model.layers[6].output)
feature_train = encoder.predict(X_train).astype('float32')
feature_train = np.reshape(feature_train, [-1,3*3*8])
feature_test = encoder.predict(X_test).astype('float32')
feature_test = np.reshape(feature_test, [-1,3*3*8])
# print (feature_train[2])
# pca = PCA(n_components=10)
# feature_train_pca = pca.fit_transform(feature_train)
# feature_test_pca = pca.fit_transform(feature_test)


clf = svm.OneClassSVM(nu=0.8, kernel="poly",gamma='auto')
clf.fit(feature_train)
y_pred_test = clf.predict(feature_test)
y_pred_train = clf.predict(feature_train)
for i in range(y_pred_train.shape[0]):
    if y_pred_train[i] == 1:
        cv2.imwrite('./cluster2/'+str(i)+'.png', (X_train[i]*255).astype('uint64'))

for i in range(y_pred_test.shape[0]):
    if y_pred_test[i] == 1:
        cv2.imwrite('./cluster3/'+str(i)+'.png', (X_test[i]*255).astype('uint64'))
