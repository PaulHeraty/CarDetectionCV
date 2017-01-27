import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import imutils

from Features import Features

class Cars:
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        self.possible_car_list = []
        # car_list = ((bbox, car_center, confidence_score, frames_tracked, frames_lost), ...)
        self.car_list = []
        self.cars_tracked = []
        self.confidence_threshold = 0.8
        self.frame_threshold = 5

    def debug_print(self, string):
        if self.debug_mode:
            print(string)

    def draw_car_rect(self, img, bb1, bb2, color, width, confidence_score, frames_tracked, frames_lost):
        if self.debug_mode == True:
            # If it's a relatively new car, then show target crosshairs
            if frames_tracked >= 0 and frames_tracked < self.frame_threshold:
                cir_x = int((bb2[0] - bb1[0])/2) + bb1[0]
                cir_y = int((bb2[1] - bb1[1])/2) + bb1[1]
                tracking_color = (255,255,0)
                cv2.circle(img, (cir_x, cir_y), 20, tracking_color, width)
                lx1 = cir_x
                lx2 = cir_x
                ly1 = cir_y - 40
                ly2 = cir_y + 40
                cv2.line(img, (lx1, ly1), (lx2, ly2), tracking_color, width)
                lx1 = cir_x - 40
                lx2 = cir_x + 40
                ly1 = cir_y 
                ly2 = cir_y 
                cv2.line(img, (lx1, ly1), (lx2, ly2), tracking_color, width)
                cv2.rectangle(img, bb1, bb2, (255,255,0), 2) 
        # Draw the surrounding box
        font = cv2.FONT_HERSHEY_SIMPLEX
        if frames_tracked >= self.frame_threshold and confidence_score >= self.confidence_threshold:
            cv2.rectangle(img, bb1, bb2, color, width) 
            # Draw confidence score
            cs = "{0:.2f}%".format(confidence_score*100.0)
            text_loc = bb1
            cv2.putText(img, cs, text_loc, font, 1, (0,255,255), 3)
        if self.debug_mode:
            if frames_tracked >= self.frame_threshold and confidence_score < self.confidence_threshold:
                cv2.rectangle(img, bb1, bb2, (255,0,0), width) 
        if self.debug_mode == True:
            # Draw frames tracked
            ft = "{}t, {}l".format(frames_tracked, frames_lost)
            text_loc = (bb1[0], bb2[1])
            cv2.putText(img, ft, text_loc, font, 1, (0,0,255), 3)
            # Draw confidence score
            cs = "{0:.2f}%".format(confidence_score*100.0)
            text_loc = bb1
            cv2.putText(img, cs, text_loc, font, 1, (0,255,255), 3)

    def draw_car_rects(self, img):
        # Make a copy of the image
        imcopy = np.copy(img)
        i = 0
        # Iterate through the boxes
        for bbox in self.cars_tracked:
            bb = bbox[0]
            confidence_score = bbox[2]
            frames_tracked = bbox[3]
            frames_lost = bbox[4]
            self.draw_car_rect(imcopy, bb[0], bb[1], (0,255,0), 6, confidence_score, frames_tracked, frames_lost) 
        return imcopy

    def get_car_center(self, bbox):
        return (int((bbox[1][0] - bbox[0][0])/2) + bbox[0][0], int((bbox[1][1] - bbox[0][1])/2) + bbox[0][1])
                 
    def check_is_same_car(self, old_car_center, new_car_center):
        old_x = old_car_center[0]
        old_y = old_car_center[1]
        new_x = new_car_center[0]
        new_y = new_car_center[1]
        delta = 64/2  # Half of smallest window size that we check for cars
 
        if new_x > old_x-delta and new_x < old_x+delta and new_y > old_y-delta and new_y < old_y+delta:
            return True
        else:
            return False

    def match_cars_with_tracked(self,  car_center):
        match = False
        matched_car = []
        # Check to see that the locality of the detected cars matches those in the history
        for car in self.car_list:
            new_car_center = car[1]
            car_tracked = car[3]
            self.debug_print("    Check tracked {} against new {}".format(car_center, new_car_center))
            if self.check_is_same_car(car_center, new_car_center) and car_tracked == 1:
                self.debug_print("    Check match")
                match = True
                matched_car = car
                break

        return match, matched_car

    def track_cars(self):
        new_cars_tracked = []
        
        if len(self.cars_tracked) == 0:
            if len(self.car_list) > 0:
                self.debug_print("Adding {} cars to empty tracked list".format(len(self.car_list)))
                self.debug_print("    {}".format(self.car_list))
                self.cars_tracked = self.car_list
        else:
            # Iterate through tracked cars to find matches
            self.debug_print("Checking {} tracked cars against {} new cars..".format(len(self.cars_tracked), len(self.car_list)))
            for car in self.cars_tracked:
                car_bbox = car[0]
                car_center = car[1]
                car_conf = car[2]
                car_tracked = car[3]
                car_lost = car[4] 
                
                # Match the cars in new frame with tracked cars
                match, matched_car =  self.match_cars_with_tracked(car_center)
                if match:
                    new_cars_tracked.append( (matched_car[0], matched_car[1], matched_car[2], car_tracked+1, 0))
                    # Use tracked value of match car in car_list to indicate that we've found a match
                    index = self.car_list.index(matched_car)
                    #self.car_list[index][3] = 0
                    this_car = list(self.car_list[index])
                    this_car[3] = 0
                    self.car_list[index] = this_car
                else:
                    new_cars_tracked.append( (car_bbox, car_center, car_conf, car_tracked, car_lost+1))
     

            # Remove tracked cars that have been lost for 5 frames or more
            for car in new_cars_tracked:
                car_tracked = car[3]
                car_lost = car[4]
                if car_lost >= 5 or car_lost >= car_tracked:
                    self.debug_print("    Removing lost car {}".format(car[1]))
                    new_cars_tracked.remove(car)

            # Add any new cars that are not already included
            for car in self.car_list:
                car_bbox = car[0]
                car_center = car[1]
                car_conf = car[2]
                car_tracked = car[3]
                if car_tracked == 1:
                    self.debug_print("    Adding un-matched new car {} to tracked list".format(car_center))
                    new_cars_tracked.append( (car_bbox, car_center, car_conf, 1, 0))

            self.debug_print("     Tracked cars : {}".format(len(new_cars_tracked)))
            self.debug_print("         {}".format(new_cars_tracked))
            self.cars_tracked = new_cars_tracked

    def resolve_bbox_overlaps(self, box1, box1_confidence_score):
        merged = False
        # Check to see if box1 overlaps any box in boxes
        for existing_car in self.car_list:
            b1x1 = box1[0][0]
            b1y1 = box1[0][1]
            b1x2 = box1[1][0]
            b1y2 = box1[1][1]
            b2x1 = existing_car[0][0][0]
            b2y1 = existing_car[0][0][1]
            b2x2 = existing_car[0][1][0]
            b2y2 = existing_car[0][1][1]
            b2cs = existing_car[2]
            #print("Checking win (({},{}),({},{})) against (({},{}),({},{}))...".format(b1x1,b1y1,b1x2,b1y2,b2x1,b2y1,b2x2,b2y2))
        
            if (b2x1 >= b1x1 and b2x1 <= b1x2) or (b2x2 >= b1x1 and b2x2 <= b1x2) or (b2x1 <= b1x1 and b2x2 >= b1x2):
                if (b2y1 >= b1y1 and b2y1 <= b1y2) or (b2y2 >= b1y1 and b2y2 <= b1y2) or (b2y1 <= b1y1 and b2y2 >= b1y2):
                    #print("Found overlap!")
                    new_box = ((min(b1x1, b2x1), min(b1y1, b2y1)), (max(b1x2, b2x2),  max(b1y2, b2y2)))
                    new_conf = max(b2cs, box1_confidence_score)
                    self.car_list.remove(existing_car)
                    self.car_list.append((new_box, self.get_car_center(new_box), new_conf, 1, 0))
                    merged = True
                    break

        # If box has not been merged, then add it as a new box
        if merged == False:
            self.car_list.append((box1, self.get_car_center(box1), box1_confidence_score, 1, 0))


    def detect_from_possible_windows(self, possible_cars):
        self.car_list = []
        # Loop through the list and merge overlaps
        for win in possible_cars:
            bbox = win[0]
            conf_score = win[1]

            # First box always passes through
            if len(self.car_list) == 0:
                self.car_list.append((bbox, self.get_car_center(bbox), conf_score, 1, 0))
                continue

            # Resolve overlaps of this box with final_bboxes
            self.resolve_bbox_overlaps(bbox, conf_score)

        # Track cars between frame
        self.track_cars()
