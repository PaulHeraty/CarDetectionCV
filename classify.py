#!/home/pedgrfx/anaconda3/bin/python3

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

from moviepy.editor import VideoFileClip
from Features import Features
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from skimage.feature import hog

test_image = True

debug_mode = False

def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image
 
	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
 
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
 
		# yield the next image in the pyramid
		yield image

def classify_boxes(img, bboxes, color=(0, 0, 255), thick=6, frame_number=0):
    # Make a copy of the image
    #imcopy = np.copy(img)
    i = 0
    bbox_list = []
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # check prediction
        #print("bbox : {}".format(bbox))
        sub_image = img[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
        # Resize the sub_image to 64x64
        sub_image_resized = cv2.resize(sub_image, (64, 64))
        # filename for debug
        filename = "./debug/frame" + str(frame_number) + "_" + str(i) 

        sub_image_f = Features(filename=filename, debug_mode=False)
        sub_image_features = sub_image_f.extract_features(sub_image_resized, cspace='YCrCb') 
        sub_image_features = sub_image_features.reshape(1, -1)

        # Use scalar to normalize feature list
        scaled_X = X_scaler.transform(sub_image_features)

        if cal.predict(scaled_X) == 1:
            #if debug_mode:
                # Draw a rectangle given bbox coordinates
                #cv2.rectangle(imcopy, bbox[0], bbox[1], (255,0,0), thick)
            # Confidence score
            #confidence_score = clf.decision_function(scaled_X)
            confidence_score = cal.predict_proba(scaled_X)
            #print(confidence_score)
            #if confidence_score[0][1] > 0.9:
                # Draw a rectangle given bbox coordinates
                #cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
                # Append window position to list
                #bbox_list.append(bbox)
            bbox_list.append((bbox, confidence_score[0][1]))
            # Dump for debug
            if debug_mode:
                filename = filename + "_" + str(confidence_score) + ".png"
                mpimg.imsave(filename, sub_image_resized)
                sub_image_f.dump_color_hist()
                sub_image_f.dump_hog_images()

        i = i + 1
    # Return the image copy with boxes drawn
    #return imcopy
    return bbox_list

def draw_car_rects(img, bboxes):
    # Make a copy of the image
    imcopy = np.copy(img)
    i = 0
    # Iterate through the boxes
    for bbox in bboxes:
        bb = bbox[0]
        confidence_score = bbox[1]
        if confidence_score >= 0.6 and confidence_score < 0.7:
            cv2.rectangle(imcopy, bb[0], bb[1], (0,0,255), 6) 
        elif confidence_score >= 0.7 and confidence_score < 0.8:
            cv2.rectangle(imcopy, bb[0], bb[1], (0,255,255), 6) 
        elif confidence_score >= 0.8 and confidence_score < 0.9:
            cv2.rectangle(imcopy, bb[0], bb[1], (0,255,0), 6) 
        elif confidence_score >= 0.9 and confidence_score < 0.95:
            cv2.rectangle(imcopy, bb[0], bb[1], (255,255,0), 6) 
        elif confidence_score >= 0.95 and confidence_score <= 1.0:
            cv2.rectangle(imcopy, bb[0], bb[1], (255,0,0), 6) 

    return imcopy
    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    #nx_windows = np.int(xspan/nx_pix_per_step) - 1
    #ny_windows = np.int(yspan/ny_pix_per_step) - 1
    nx_windows = np.int((xspan-xy_window[0])/nx_pix_per_step)
    ny_windows = np.int((yspan-xy_window[1])/ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    #     Note: you could vectorize this step, but in practice
    #     you'll be considering windows one by one with your
    #     classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define pipeline
def process_image(img):
    possible_cars = []
    # Run different size sliding windows over image
    windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[380, 700], 
                        xy_window=(256, 256), xy_overlap=(0.9, 0.9))
    print(len(windows))
    possible_cars = possible_cars + classify_boxes(img, windows, color=(0, 0, 255), thick=6)
    #print(len(possible_cars))
    windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[380, 600], 
                        xy_window=(128, 128), xy_overlap=(0.85, 0.85))
    print(len(windows))
    possible_cars = possible_cars + classify_boxes(img, windows, color=(0, 0, 255), thick=6)
    #print(len(possible_cars))
    windows = slide_window(img, x_start_stop=[300, 1000], y_start_stop=[380, 480], 
                        xy_window=(64, 64), xy_overlap=(0.75, 0.75))
    print(len(windows))
    possible_cars = possible_cars + classify_boxes(img, windows, color=(0, 0, 255), thick=6)                    
    #print(len(possible_cars))
    windows = slide_window(img, x_start_stop=[400, 900], y_start_stop=[380, 480], 
                        xy_window=(32, 32), xy_overlap=(0.5, 0.5))
    print(len(windows))
    possible_cars = possible_cars + classify_boxes(img, windows, color=(0, 0, 255), thick=6)                    
    #print(len(possible_cars))
    #windows = slide_window(img, x_start_stop=[500, 900], y_start_stop=[380, 480], 
    #                    xy_window=(16, 16), xy_overlap=(0.5, 0.5))
    #print(len(windows))
    #possible_cars = possible_cars + classify_boxes(img, windows, color=(0, 0, 255), thick=6)                    

    # Draw the windows
    window_img = draw_car_rects(img, possible_cars)    

    # Draw checked areas for debug
    #cv2.rectangle(window_img, (0,380), (1280,700), (0,0,255), 6) 
    #cv2.rectangle(window_img, (0,380), (1280,600), (0,0,255), 6) 
    #cv2.rectangle(window_img, (300,380), (1000,480), (0,0,255), 6) 
    #cv2.rectangle(window_img, (400,380), (900,480), (0,0,255), 6) 
    #cv2.rectangle(window_img, (500,380), (900,480), (0,0,255), 6) 

    return window_img
                           
# Load the classifier model
clf = joblib.load('./models/classifier.pkl')
cal = joblib.load('./models/calibrated.pkl')

# Load the standard scalar model
X_scaler = joblib.load('./models/scaler.pkl')

if test_image:
    print("Running on test images...")
    #####################################
    # Run our pipeline on the test images
    #####################################
    images = glob.glob("/home/pedgrfx/SDCND/CarDetection/test_images/test*.jpg")
    images.sort()

    # Setup the plot grid for test images
    plt.figure(figsize = (len(images),2))
    gs1 = gridspec.GridSpec(len(images),2)
    gs1.update(wspace=0.025, hspace=0.05)
    i=0

    for fname in images:
        print("Processing image {}".format(fname))

        # Next, let's read in a test image
        img = mpimg.imread(fname)

        # Process the image using our pipeline
        combined_img = process_image(img)
    
        # Plot the original image and the processed images
        ax1 = plt.subplot(gs1[i])
        plt.axis('off')
        ax1.imshow(img)
        ax2 = plt.subplot(gs1[i+1])
        plt.axis('off')
        ax2.imshow(combined_img)
        i += 2
    plt.show()

else: # test video
    print("Running on test video1...")
    #####################################
    # Run our pipeline on the test video 
    #####################################
    clip = VideoFileClip("./project_video.mp4")
    output_video = "./project_video_processed.mp4"
    output_clip = clip.fl_image(process_image)
    output_clip.write_videofile(output_video, audio=False)


    
