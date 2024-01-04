import os
from cProfile import label

import labels
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random

from contourpy.util import data
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC



# dir = 'C:\Users\harsh doshi\OneDrive\Desktop\dogs-vs-cats\test1'
#
# categories = ["Cat", "Dog"]
#
# for category in categories:
#     path = os.path.join(dir, category)
#
#     for img in os.listdir(path):
#         imgpath = os.path.join(path, img)
#         pet_img = cv2.imread(imgpath, 0)
#         try:
#             pet_img = cv2.resize(pet_img,(50,50))
#             image = np.array(pet_img).flatten()
#
#             data.append([image,label])
#         except Exception as e:
#             pass
#
#
#
# print(len(data))
#
# pick_in = open('data1.pickle', 'wb')
# pickle.dumb('data1.pickle','wb')
# pick_in.close()
pick_in = open('data1.pickle', 'wb')
data=pickle.load(pick_in)
pick_in.close()

random.shuffle(data)
features = []
label = []

for feature, label, in data:
    features.append(feature)
    labels.append(label)


xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.50)

# model = SVC(C=1, kernel='poly' ,gamma= 'auto')
# model.fit(xtrain, ytrain)

pick = open('model.sav','rb')
model = pickle.load(pick)
pick.close()

prediction=model.predict(xtest)
accuracy = model.scope(xtest,ytest)

print('Accuracy', accuracy)

print('Prediction is: ', categories[prediction[0]])


mypet=xtest[0].reshape(50,50)
plt.show(mypet,cmap='gray')
plt.show()
