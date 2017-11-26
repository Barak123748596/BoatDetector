import urllib
import cv2
from config import *
import argparse
import pySaliencyMap
import numpy as np
import time
import os
import math
from geopy.distance import vincenty

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-lo", "--longtitude", help="longtitude", type=float)
    parser.add_argument("-la", "--latitude", help="latitude", type=float)
    parser.add_argument("-z", "--zoom", help="zoom scale, default 18", type=int, default=18) 
    parser.add_argument("-m", "--mode", help="binary converting mode, 1: kmean, 2: saliency, 3: combine", type=int, choices=xrange(1, 4))
    args = parser.parse_args()
    return args

def downloadImage(longtitude, latitude, zoom_scale, key):
    try:
        print("Downloading...")
        DIR = "{}_{}".format(longtitude, latitude)
        if not os.path.exists(DIR):
            os.mkdir(DIR)
        os.chdir(DIR)
        satellite_file="satellite.png"
        mask_file="mask.png"
        urllib.urlretrieve(SATELLITE_DOWNLOAD_URL.format(longtitude,latitude, zoom_scale, key), satellite_file)
        urllib.urlretrieve(MASK_DOWNLOAD_URL.format(longtitude,latitude, zoom_scale, key), mask_file)
    except:
        print("Failed!")
        return None
    else:
        print("Successful!")
        return satellite_file, mask_file

def toBinarySaliency(image):
    imgsize = image.shape
    img_width  = imgsize[1]
    img_height = imgsize[0]
    sm = pySaliencyMap.pySaliencyMap(img_width, img_height)
    binarized_map = sm.SMGetBinarizedSM(cv2.medianBlur(image, 15))
    cv2.imwrite("satellite_2b_bin_saliency.jpg", binarized_map)
    return binarized_map
def brightness(rgb):
    return rgb[0]*0.299 + rgb[1]*0.587 + rgb[2]*0.114

def toBinaryKmean(image):
    # Kmean
    Z = image.reshape((-1,3))
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    if brightness(center[0]) > brightness(center[1]):
        center[0]=[255,255,255]
        center[1]=[0,0,0]
    else:
        center[1]=[255,255,255]
        center[0]=[0,0,0]
    img = center[label.flatten()]
    kmean = img.reshape((image.shape))
    cv2.imwrite("satellite_1_kmean.jpg", kmean)
    # Binary
    thresh=cv2.cvtColor(kmean, cv2.COLOR_BGR2GRAY)
    # Noise filter
    kernel = np.ones((5,5),np.float32)/25
    filted = cv2.filter2D(thresh,-1,kernel)
    filted[filted>=127]=255
    filted[filted<127]=0

    # Dilation
    kernel = np.ones((20,20),np.uint8)
    dilation = cv2.dilate(filted,kernel,iterations = 1)
    cv2.imwrite("satellite_2a_bin_kmean.jpg", dilation)
    return dilation

def toBinaryCombine(image):
    binary_saliency = toBinarySaliency(image)
    binary_kmean = toBinaryKmean(image)

    binary_combine = cv2.bitwise_or(binary_saliency, binary_kmean)
    cv2.imwrite("satellite_2c_bin_combine.jpg", binary_combine)
    return binary_combine

def extractWater(satellite, mask):
    satellite=satellite[:-50,:]
    origin_image =satellite.copy()
    mask=mask[:-50,:]

    mask[mask>0]=255
    inv_mask=255-mask
    inv_mask=inv_mask.astype('bool')
    mean=np.mean(satellite[inv_mask],axis=0)

    bool_mask = mask.astype('bool')
    satellite[bool_mask] = mean
    cv2.imwrite("satellite_0_water_remove.jpg", satellite)
    return satellite, origin_image

def detectContour(binary_image, origin_image, center, args):
    log = args.longtitude
    lat = args.latitude
    zoom =args.zoom

    _, contours,_ = cv2.findContours(binary_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    img = origin_image.copy()
    count=0
    for cnt in contours:
        try:
            area = cv2.contourArea(cnt)
            (X,Y),(ma,Ma),angle = cv2.fitEllipse(cnt)
            if area>500:
                x,y,W,H = cv2.boundingRect(cnt)
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                crop = crop_minAreaRect(origin_image, rect)
                w,h,_ = crop.shape

                if (max(w,h)*1.0/min(w,h) > 2):
                    img = cv2.drawContours(img,[box],-1,(0,255,0),2)
                    
                    dist_x = x+ W/2 -center[0]
                    dist_y = y+ H/2 -center[1]

                    center_latitude = lat+dist_x/ math.pow(2, zoom+1)
                    center_longtitude = log-dist_y/math.pow(2,zoom+1)

                    cv2.putText(img,"({},{})".format(center_longtitude,center_latitude),(x,y+H/2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2)
                    img = cv2.circle(img,(x+W/2,y+H/2), 5, (0,0,255), -1)
                    
                    dist_x = x -center[0]
                    dist_y = y -center[1]

                    corner_latitude = lat+dist_x/ math.pow(2, zoom+1)
                    corner_longtitude = log-dist_y/math.pow(2,zoom+1)

                    length = 2*vincenty((center_longtitude,center_latitude), (corner_longtitude, corner_latitude)).meters             
                    cv2.putText(img,"length:{}".format(length),(x,y+H), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2)
                    
                    name = "{}.png".format(count)
                    cv2.imwrite(name, crop)
                    print(name+" saved")
                    count+=1
        except Exception as e:
            print(e)
    cv2.imwrite("satellite_3_result.png",img)

def crop_minAreaRect(img, rect):
    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]    
    pts[pts < 0] = 0

    # crop
    y1 = max(pts[1][1]-20, 0)
    y2 = min(pts[0][1]+20, rows)
    x1 = max(pts[1][0]-20, 0)
    x2 = min(pts[2][0]+20, cols)
    img_crop = img_rot[y1:y2,x1:x2]

    return img_crop

def main():
    args = arg_parse()
    save_files = downloadImage(args.longtitude, args.latitude, args.zoom, KEY)
    if save_files is None:
        print("download error")
        return False
    satellite = cv2.imread(save_files[0])
    center = (satellite.shape[0]/2,satellite.shape[1]/2)
    mask = cv2.imread(save_files[1],0)
   
    print("Water extracting...\n")
    extracted_water, origin_image = extractWater(satellite, mask)
    # os.remove(save_files[0])
    # os.remove(save_files[1])

    print("Binarizing image...\n")
    binary_img = np.zeros(extracted_water.shape[:2])
    if args.mode == 1:
        binary_img = toBinaryKmean(extracted_water)
    elif args.mode == 2:
        binary_img = toBinarySaliency(extracted_water)
    elif args.mode == 3:
        binary_img = toBinaryCombine(extracted_water)
    binary_img = binary_img.astype(np.uint8)
    
    print("Detecting object...\n")
    detectContour(binary_img, origin_image, center, args)

if __name__ == '__main__':
    main()



        





                    