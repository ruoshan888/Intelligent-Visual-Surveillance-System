import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk 
from skimage.color import label2rgb
plt.close('all')

fgModel = cv2.createBackgroundSubtractorMOG2()
leastNumOfFrames = 5
idx = []
C = []

video_to_process = "car.mp4"

#%% Create folder to store frame and binary segmented image
filename = os.path.splitext(video_to_process)[0]
print(filename)
folder_out1 = filename + '_frame'
folder_out2 = filename + '_binary'

if not os.path.exists(folder_out1):
    os.makedirs(folder_out1)

if not os.path.exists(folder_out2):
    os.makedirs(folder_out2)    

#%%
captured_video = cv2.VideoCapture(video_to_process)

i=0
while True:
    # read video frames
    retval, frame = captured_video.read()

    # check whether the frames have been grabbed
    if not retval:
        break

    # resize video frames
    frame = cv2.resize(frame, (640, 360))

    # pass the frame to the background subtractor
    fgmask = fgModel.apply(frame)
 
 
 
 
 
 
 
    # Morphological operation settings
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    
    # Apply morphological operations to clean up noise
    fgmask_cleaned = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    fgmask_cleaned = cv2.morphologyEx(fgmask_cleaned, cv2.MORPH_OPEN, kernel)
     
    # Convert cleaned mask to binary using adaptive thresholding (Otsu's method)
    _, binary_image = cv2.threshold(fgmask_cleaned, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Do connected component labeling  
    label_image = label(binary_image, connectivity=2, background=0)
    
    # Iterate over detected objects and filter them based on size (people size range)
    for region in regionprops(label_image):
        if region.area >= 1000:  # Filter out small noise objects
            # Get the bounding box coordinates
            minr, minc, maxr, maxc = region.bbox
            width = maxc - minc
            height = maxr - minr
            
            
            # Adjust the classification logic to better distinguish between cars and persons
            
            if region.area > 8000:
                object_type = "Car"
            elif region.area > 5000:
                object_type = "Motorcycle"
            else : object_type = "Person"
            

            # Draw bounding box around detected object
            cv2.rectangle(frame, (minc, minr), (maxc, maxr), (0, 255, 0), 2)

            # Display the label (Person/Vehicle)
            cv2.putText(frame, object_type, (minc, minr - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)






    # show the current frame, foreground mask, subtracted result
    cv2.imshow("Frames", frame)
    # show segmented foreground as binary image 
    cv2.imshow("Foreground Masks", fgmask_cleaned)
    
    file_frame = os.path.join(folder_out1, 'Frame'+str(i) +'.jpg')
    cv2.imwrite(file_frame, frame)
    file_frame = os.path.join(folder_out2, 'Frame'+str(i) +'.jpg')
    cv2.imwrite(file_frame, fgmask_cleaned)
    
    i = i+1
    keyboard = cv2.waitKey(10)
    if keyboard == 27:
        break

captured_video.release() 
cv2.destroyAllWindows() 