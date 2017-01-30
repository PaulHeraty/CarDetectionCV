import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle
import imutils
import matplotlib.gridspec as gridspec
import os
import ntpath

import scipy.misc as scimsc
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from skimage.feature import hog

class Features:
    def __init__(self, filename="", debug_mode=False):
        self.debug_mode =  debug_mode
        self.filename = filename
        self.channel1_hist = []
        self.channel2_hist = []
        self.channel3_hist = []
        self.hog_imageR = []
        self.hog_imageG = []
        self.hog_imageB = []

    def bin_spatial(self, img, size=(32, 32)):
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, size).ravel() 
        # Return the feature vector
        return features

    # Define a function to compute color histogram features  
    def color_hist(self, img, nbins=32, bins_range=(0, 256)):
        # Compute the histogram of the color channels separately
        self.channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        self.channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        self.channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((self.channel1_hist[0], self.channel2_hist[0], self.channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    def dump_color_hist(self):
        bin_edges = self.channel1_hist[1]
        bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
        fig = plt.figure(figsize=(12,3))
        plt.subplot(131)
        plt.bar(bin_centers, self.channel1_hist[0])
        plt.xlim(0, 256)
        plt.title('Channel 1')
        plt.subplot(132)
        plt.bar(bin_centers, self.channel2_hist[0])
        plt.xlim(0, 256)
        plt.title('Channel 2')
        plt.subplot(133)
        plt.bar(bin_centers, self.channel3_hist[0])
        plt.xlim(0, 256)
        plt.title('Channel 3')
        fig.tight_layout()
        plt.savefig(self.filename + "_color_hist.png")
        plt.close()


    # Define a function to return HOG features and visualization
    def get_hog_features(self, img, orient, pix_per_cell, cell_per_block, 
                        feature_vec=True):
        # Call with two outputs if vis==True
        if self.debug_mode == True:
            featuresR, self.hog_imageR = hog(img[:,:,0], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=True, feature_vector=feature_vec)
            featuresG, self.hog_imageG = hog(img[:,:,1], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=True, feature_vector=feature_vec)
            featuresB, self.hog_imageB = hog(img[:,:,2], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=True, feature_vector=feature_vec)
            return featuresR, featuresG, featuresB 
        # Otherwise call with one output
        else:      
            featuresR = hog(img[:,:,0], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=False, feature_vector=feature_vec)
            featuresG = hog(img[:,:,1], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=False, feature_vector=feature_vec)
            featuresB = hog(img[:,:,2], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=False, feature_vector=feature_vec)
            return featuresR, featuresG, featuresB

    def dump_hog_images(self):
        filename = self.filename + "_hog_1.png"
        mpimg.imsave(filename, self.hog_imageR)
        filename = self.filename + "_hog_2.png"
        mpimg.imsave(filename, self.hog_imageG)
        filename = self.filename + "_hog_3.png"
        mpimg.imsave(filename, self.hog_imageB)

    # Define a function to extract features from a list of images
    # Have this function call bin_spatial() and color_hist()
    def extract_features(self, image, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256), orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
        # If image is RGBA, conver to RGB
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'LAB':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
        else: feature_image = np.copy(image)      
        # Apply bin_spatial() to get spatial color features
        #spatial_features = self.bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = self.color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        # Call get_hog_features() with vis=False, feature_vec=True
        hog_features_R, hog_features_G, hog_features_B = self.get_hog_features(feature_image, orient, 
                        pix_per_cell, cell_per_block, feature_vec=True)
        # Append the new feature vector to the features list
        #features = np.concatenate((spatial_features, hist_features, hog_features_R, hog_features_G, hog_features_B))
        #features = np.concatenate((spatial_features, hist_features, hog_features))
        features = np.concatenate((hist_features, hog_features_R, hog_features_G, hog_features_B))
        #features = np.concatenate((hog_features_R, hog_features_G, hog_features_B))

        # Return list of feature vectors
        return features

    def extract_image_file_features(self, imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256), orient=9, 
                        pix_per_cell=8, cell_per_block=2 ):

        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in imgs:
            # Read in each one by one
            image = scimsc.imread(file)
            
            image_features = self.extract_features(image, cspace=cspace, spatial_size=spatial_size,
                        hist_bins=hist_bins, hist_range=hist_range, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block )
            
            features.append(image_features)

        return features

