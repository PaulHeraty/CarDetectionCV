#!/home/pedgrfx/anaconda3/bin/python3

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle
from Features import Features
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.cross_validation import train_test_split

# All images are 64x64 pixels
cars = glob.glob('./vehicles/*/*png')
notcars = glob.glob('./non-vehicles/*/*png')

print("Found {} car images".format(len(cars)))
print("Found {} non-car images".format(len(notcars)))

print("Extrating feature vectors from car images...")
car_f= Features()
car_features = car_f.extract_image_file_features(cars, cspace='YCrCb')
print("Extrating feature vectors from notcar images...")
notcar_f= Features()
notcar_features = notcar_f.extract_image_file_features(notcars, cspace='YCrCb')

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

# Use a linear SVC 
svc = LinearSVC(C=1.0)
# Add a calibrated classifier to get probabilities
clf = CalibratedClassifierCV(svc)
# Check the training time for the SVC
t=time.time()
print("Training Linear SVC Classifier...")
clf.fit(X_train, y_train)
t2 = time.time()
print(t2-t, 'Seconds to train SVC...')
# Check the score of the SVC
print('Train Accuracy of SVC = ', clf.score(X_train, y_train))
print('Test Accuracy of SVC = ', clf.score(X_test, y_test))
# Check the prediction time for a single sample
t=time.time()
prediction = clf.predict(X_test[0].reshape(1, -1))
prob = clf.predict_proba(X_test[0].reshape(1, -1))
t2 = time.time()
print(t2-t, 'Seconds to predict with SVC')
print("Prediction {}".format(prediction))
print("Prob {}".format(prob))

# Save model & scaler
print("Saving models...")
joblib.dump(svc, './models/classifier.pkl')
joblib.dump(clf, './models/calibrated.pkl')
joblib.dump(X_scaler, './models/scaler.pkl')
print("Done!")
