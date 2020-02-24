#%% detect cup

import argparse
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.ndimage as ndimage
from matplotlib import pyplot as plt
from skimage import exposure, filters
from skimage.draw import circle_perimeter
from skimage.feature import canny
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse


def resize(img):
    width = 1024
    height = 720
    #####
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)


def rgb2Blue(img):
    b, g, r = cv2.split(img)
    return b


def rgb2Red(img):
    b, g, r = cv2.split(img)
    return r


def rgb2Green(img):
    b, g, r = cv2.split(img)
    return g


def rgb2Gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def rgb2lab(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)


#############preprocess###########
##Image is split on B,G,R channel
##Red channel is isolated
##Smoothing over the red channel is applied
##Sharpening and Equalization to te image are applied
##A morph closing is applied to remove artifacts
##################################
def preprocess(img):
    b, g, r = cv2.split(img)
    gray = rgb2Red(img)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.addWeighted(gray, 1.5, gray_blur, -0.5, 0, gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    gray = ndimage.grey_closing(gray, structure=kernel)
    gray = cv2.equalizeHist(gray)
    # gray = cv2.GaussianBlur(gray, (5,5), 0)
    # gray = cv2.medianBlur(gray,9)

    # gray_la= ndimage.laplace(gray)
    # gray = cv2.GaussianBlur(gray,(3,3),0)
    # gray_eq = cv2.equalizeHist(gray)

    return gray


#############getROI##############
##Image is resized
##We take green channel and smooth it
##Opening is done to remove artifacts, in order to preserve only BRIGHTEST elements
##Now we get the most bright pixel position
##We return that position in a 110x110 window
##It is actually a simple way to detect the optic disc, but it works so..
##################################
def getROI(image):
    image_resized = image #resize(image)
    b, g, r = cv2.split(image_resized)
    g = cv2.GaussianBlur(g, (15, 15), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    g = ndimage.grey_opening(g, structure=kernel)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(g)

    x0 = int(maxLoc[0]) - 90
    y0 = int(maxLoc[1]) - 90
    x1 = int(maxLoc[0]) + 90
    y1 = int(maxLoc[1]) + 90

    return  x0,x1,y0,y1


def getValue(img):
    shapeRow = img.shape[0]
    shapeCol = img.shape[1]
    x = 0
    y = 0
    acu = 0
    maxloc = []
    for i in range(shapeRow):
        for j in range(shapeCol):
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(img[i - 15:j - 15, i + 15:j + 15])
            value = maxVal
            if value > acu:
                acu = value
                maxloc = maxLoc
    return maxloc


def kmeans(img):
    ## K-Means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    roi = img
    X = roi.reshape((-1, 1))
    X = np.float32(X)
    compactness, labels, centers = cv2.kmeans(X, 3, None, criteria, 10, flags)

    result = np.choose(labels, centers)
    result.shape = X.shape

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    res2 = res.reshape((roi.shape))
    return res2


def checkSide(img):
    shapeRow = img.shape[0]
    shapeCol = img.shape[1]
    if cv2.countNonZero(img[:, 0:int(shapeCol / 2)]) > cv2.countNonZero(img[:, int(shapeCol / 2):shapeCol]):
        return True
    else:
        return False


def checkHigh(img):
    shapeRow = img.shape[0]
    shapeCol = img.shape[1]
    if cv2.countNonZero(img[0:int(shapeRow / 2), :]) > cv2.countNonZero(img[int(shapeRow / 2):shapeRow, :]):
        return True
    else:
        return False


def canny(img, sigma):
    v = np.mean(img)
    sigma = sigma
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(img, lower, upper)
    return edged


def hough(edged, limm, limM):
    hough_radii = np.arange(limm, limM, 1)
    hough_res = hough_circle(edged, hough_radii)
    return hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)


def detect_cup(filepath):

    image = cv2.imread(filepath)
    roi = getROI(image)
    #preprocessed_roi = preprocess(roi)
    
    #print(roi)
    
    return roi



#%%

from PIL import Image, ImageDraw
import numpy as np
from joblib import Parallel, delayed
from skimage.morphology import skeletonize
import random
import math


def draw_ellipse(image, bounds, width=1, outline='white', antialias=4):
    """Improved ellipse drawing function, based on PIL.ImageDraw."""

    # Use a single channel image (mode='L') as mask.
    # The size of the mask can be increased relative to the imput image
    # to get smoother looking results. 
    mask = Image.new(
        size=[int(dim * antialias) for dim in image.size],
        mode='L', color='black')
    draw = ImageDraw.Draw(mask)

    # draw outer shape in white (color) and inner shape in black (transparent)
    for offset, fill in (width/-2.0, 'white'), (width/2.0, 'black'):
        left, top = [(value + offset) * antialias for value in bounds[:2]]
        right, bottom = [(value - offset) * antialias for value in bounds[2:]]
        draw.ellipse([left, top, right, bottom], fill=fill)

    # downsample the mask using PIL.Image.LANCZOS 
    # (a high-quality downsampling filter).
    mask = mask.resize(image.size, Image.LANCZOS)
    # paste outline color to input image through the mask
    image.paste(outline, mask=mask)


###For visualize
def enlarge_skeleton(skeleton):

    i = 1

    if skeleton.shape[-1] == 3:
        new_skeleton = np.zeros(skeleton.shape, dtype=np.uint8)
        for row_id, row in enumerate(skeleton):
            for col_id, elem in enumerate(row):
                if elem[0] == 0 and elem[1] == 0 and elem[2] == 0:
                    new_skeleton[row_id, col_id] = (255,255,255)
                    continue

                for x in range(-i, i+1):
                    if row_id+x >= skeleton.shape[0] or row_id+x < 0:
                        continue
                    for y in range(-i, i+1):
                        if col_id+y >= skeleton.shape[1] or col_id+y < 0:
                            continue
                        new_skeleton[row_id+x, col_id+y] = elem
    else:
        new_skeleton = np.zeros((skeleton.shape[0],skeleton.shape[1],3), dtype=np.uint8)
        for row_id, row in enumerate(skeleton):
            for col_id, elem in enumerate(row):
                if elem == 0:
                    new_skeleton[row_id, col_id] = (255, 255, 255)
                    continue

                for x in range(-i, i+1):
                    if row_id+x >= skeleton.shape[0] or row_id+x < 0:
                        continue
                    for y in range(-i, i+1):
                        if col_id+y >= skeleton.shape[1] or col_id+y < 0:
                            continue
                        new_skeleton[row_id+x, col_id+y] = (0, 0, 255)

    return new_skeleton


#%%
pred_vessel = Image.open('temp/seg.png')
pred_vessel = np.array(pred_vessel, dtype=np.uint8)

vis_seg = np.zeros([pred_vessel.shape[0],pred_vessel.shape[1],3], dtype=np.uint8)
for row_id, row in enumerate(pred_vessel):
    for col_id, elem in enumerate(row):
        vis_seg[row_id, col_id] = [255-elem, 255, 255-elem]
Image.fromarray(vis_seg, 'RGB').save('temp/vessel_vis.png')


pred_vessel_binary = (pred_vessel > 120).astype(np.uint8)
pred_skeleton = (skeletonize(pred_vessel_binary)*255).astype(np.uint8)
pred_skeleton_img = Image.fromarray(enlarge_skeleton(pred_skeleton), 'RGB')#.save('temp/skeleton_vessel_vis.png')

cup = detect_cup('temp/raw.png')
cup = [cup[2], cup[0], cup[3], cup[1]]
cup_point = [int((cup[0]+cup[2])/2), int((cup[1]+cup[3])/2)]
print(cup)

r = 1 

crossing_points = []
end_points = []
single_points = []

pred_skeleton_points = pred_skeleton.copy()


segments = []

def is_img_empty(array):
    for row in array:
        for elem in row:
            if elem:
                return False
    return True

def in_boundary(point, boundaries):
    for boundary in boundaries:
        if abs(point[0]-boundary[0]) + abs(point[1]-boundary[1]) < 2:
            return False
    return True

def cal_distance(point_1, point_2):
    vector = [point_1[0] - point_2[0], point_1[1] - point_2[1]]
    distance = math.sqrt(vector[0] ** 2 + vector[1] ** 2)

    return distance

def cal_direction(p1, p2):
    vector = [p1[0] - p2[0], p1[1] - p2[1]]
    norm = cal_distance(p1, p2)
    direction = [vector[0] / norm, vector[1] / norm]

    return direction

def cal_angle(v1, v2):
    return math.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))

pre_segment_num = -1
run_count = 0
while (not is_img_empty(pred_skeleton_points)) and (not len(segments) == pre_segment_num):
    run_count+=1
    pre_segment_num = len(segments)
    for row_id, row in enumerate(pred_skeleton_points):
        final_row = []
        for col_id, elem in enumerate(row):
            if not elem:
                continue
            neighbours = []
            for x in range(-1,2):
                for y in range(-1,2):
                    if not pred_skeleton_points[row_id+y, col_id+x]:
                        continue
                    if x==0 and y==0:
                        continue
                    neighbours.append([x,y])
            
            duplicate = 0
            for neighbour in neighbours:
                if [neighbour[0]+1,neighbour[1]] in neighbours:
                    duplicate += 1
                if [neighbour[0]-1,neighbour[1]] in neighbours:
                    duplicate += 1
                if [neighbour[0],neighbour[1]+1] in neighbours:
                    duplicate += 1
                if [neighbour[0],neighbour[1]-1] in neighbours:
                    duplicate += 1
            
            neighbours_num = len(neighbours) - duplicate/2

            for neighbour in neighbours:
                if [neighbour[0]+1,neighbour[1]] in neighbours \
                    and [neighbour[0],neighbour[1]+1] in neighbours :
                    neighbours_num += 1
                if [neighbour[0]+1,neighbour[1]] in neighbours \
                    and [neighbour[0],neighbour[1]-1] in neighbours :
                    neighbours_num += 1
                if [neighbour[0]-1,neighbour[1]] in neighbours \
                    and [neighbour[0],neighbour[1]+1] in neighbours :
                    neighbours_num += 1
                if [neighbour[0]-1,neighbour[1]] in neighbours \
                    and [neighbour[0],neighbour[1]-1] in neighbours :
                    neighbours_num += 1
                    
            if neighbours_num > 2:
                crossing_points.append([row_id, col_id])
            elif neighbours_num == 1:
                end_points.append([row_id, col_id])
            elif neighbours_num == 0:
                single_points.append([row_id, col_id])
                pred_skeleton_points[row_id, col_id] = 0

    print(len(crossing_points),len(end_points),len(single_points))
    



    start_points = end_points + crossing_points
    
    i = 0
    while i < len(start_points):
        # for start_point in start_points:
        #     pred_skeleton_points[start_point[0],start_point[1]] = 255

        finish_curr_start_point = False
        # print(i, len(start_points))
        segment = {'points':[], 'endpoints':[]}

        start_point = start_points[i]

        curr_point = start_point
        segment['endpoints'].append(curr_point)
        # count = 0
        while curr_point:
            # if i == 192:
            #     count += 1
            # if count>3:
            #     aslkjfalsfdljk()
            # print(curr_point)
            pred_skeleton_points[curr_point[0], curr_point[1]] = 0
            segment['points'].append(curr_point)
            neighbouring_end = False
            curr_point_temp = None
            boundaries = []
            for x in range(-1,2):
                for y in range(-1,2):
                    # print(segment)
                    if x==0 and y==0:
                        continue
                    # if i == 192:
                    # print(3)
                    if not [curr_point[0]+y, curr_point[1]+x] in segment['endpoints']\
                        and [curr_point[0]+y, curr_point[1]+x] in start_points:
                        if len(segment['points']) > 1:
                            # print([curr_point[0]+y, curr_point[1]+x])
                            neighbouring_end = True
                            pred_skeleton_points[curr_point[0]+y, curr_point[1]+x] = 0
                            end_point = [curr_point[0]+y, curr_point[1]+x]
                            break
                        else:
                            pred_skeleton_points[curr_point[0]+y, curr_point[1]+x] = 0
                            boundaries.append([curr_point[0]+y, curr_point[1]+x])
                            if curr_point_temp and not in_boundary(curr_point_temp, boundaries):
                                curr_point_temp = None
                            continue

                    # if i == 192:
                    # print(1)
                    if not pred_skeleton_points[curr_point[0]+y, curr_point[1]+x]:
                        continue

                    if not in_boundary([curr_point[0]+y, curr_point[1]+x], boundaries):
                        continue
                    # if i == 192:
                    # print(2)
                    curr_point_temp = [curr_point[0]+y, curr_point[1]+x]
            # print(curr_point_temp, neighbouring_end, curr_point_temp in start_points)
            if curr_point_temp:
                curr_point = curr_point_temp
            else:
                segment['endpoints'].append(curr_point)
                curr_point = None
                finish_curr_start_point = True
            
            if neighbouring_end and not finish_curr_start_point:
                # if i == 192:
                #     print(4)
                # print('end point: ', end_point)
                segment['endpoints'].append(end_point)
                segment['points'].append(end_point)
                pred_skeleton_points[end_point[0], end_point[1]] = 0
                curr_point = None
        if len(segment['points']) > 1:
            # if len(segment['points']) == 2 \
            #     and segment['endpoints'][0] in start_points \
            #     and segment['endpoints'][1] in start_points:
            # finish_curr_start_point
            segments.append(segment)
            point_id = int(len(segment['points']) / 2)
            # print(segment['thickness'])
        # print('segemnts_num: ', len(segments))
        if finish_curr_start_point:
            i += 1
    print(len(segments))





crossing_points = []
for segment in segments:
    for endpoint_id in range(0,2):
        nes = []
        endpoint = segment['endpoints'][endpoint_id]

        count = 0
        nes = []
        for segment_for_search in segments:
            if len(segment_for_search['points']) < 3:
                continue
            for endpoint_id_for_search in range(0,2):
                endpoint_for_search = segment_for_search['endpoints'][endpoint_id_for_search]
                if endpoint == endpoint_for_search or \
                    cal_distance(end_point, endpoint_for_search) < 2:
                   count += 1
                   points_num = int(len(segment_for_search['points'])/2)
                   nes.append(cal_direction(segment_for_search['points'][points_num], endpoint))
        k = 0
        for line1 in nes:
            counted = []
            for line2 in nes:
                if line1  == line2:
                    continue
                if line2 in counted:
                    continue
                if cal_angle(line1, line2)<20:
                    counted.append(line1)
                    counted.append(line2)
                    k += 1
                    break
                

        if len(nes) > 4:
            crossing_points.append(endpoint)
print(len(crossing_points))

def find_av(cls_array, cls_point):
    a = False
    v = False
    i = 10
    for x in range(-i, i+1):
        for y in range(-i, i+1):
            X,Y = (cls_point[1]+x, cls_point[0]+y)
            if cls_array[X, Y, 0] > cls_array[X, Y, 2]:
                a = True
            elif cls_array[X, Y, 2] > cls_array[X, Y, 0]:
                v = True
            
            if a and v:
                return True

    return False

cls_array = np.array(Image.open('temp/cls.jpg'), dtype=np.uint8)
r = 5
pred_skeleton_img_edit1 = Image.fromarray(enlarge_skeleton(np.array(pred_skeleton_img, dtype=np.uint8)), 'RGB')
draw1 = ImageDraw.Draw(pred_skeleton_img_edit1)
for point in crossing_points:
    cls_point = cls_array[point[0],point[1]]
    if find_av(cls_array, cls_point):
        draw1.ellipse((point[1]-r, point[0]-r, point[1]+r, point[0]+r), fill=(255,205,89))
draw1.ellipse((cup_point[1]-3*r, cup_point[0]-3*r, cup_point[1]+3*r, cup_point[0]+3*r), fill=(255,0,255))
# draw1.ellipse((cup[1], cup[0], cup[3], cup[2]), fill=None, outline=(255,0,255))
draw_ellipse(pred_skeleton_img_edit1, (cup[1], cup[0], cup[3], cup[2]), outline=(255,0,255), width=r, antialias=8)
pred_skeleton_img_edit1.save('temp/skeleton_vessel_points_vis.png')
r = 1

r = 5
pred_skeleton_img_edit1 = Image.open('temp/raw.png')
draw1 = ImageDraw.Draw(pred_skeleton_img_edit1)
draw1.ellipse((cup_point[1]-3*r, cup_point[0]-3*r, cup_point[1]+3*r, cup_point[0]+3*r), fill=(255,0,255))
# draw1.ellipse((cup[1], cup[0], cup[3], cup[2]), fill=None, outline=(255,0,255))
draw_ellipse(pred_skeleton_img_edit1, (cup[1], cup[0], cup[3], cup[2]), outline=(255,0,255), width=r, antialias=8)
pred_skeleton_img_edit1.save('temp/skeleton_vessel_points_vis1.png')
r = 1
dsfsdfdf()


r = 5
pred_skeleton_img_edit1 = Image.fromarray(enlarge_skeleton(np.array(pred_skeleton_img, dtype=np.uint8)), 'RGB')
draw1 = ImageDraw.Draw(pred_skeleton_img_edit1)
for point in crossing_points:
    draw1.ellipse((point[1]-r, point[0]-r, point[1]+r, point[0]+r), fill=(255,205,89))
draw1.ellipse((cup_point[1]-3*r, cup_point[0]-3*r, cup_point[1]+3*r, cup_point[0]+3*r), fill=(255,0,255))
# draw1.ellipse((cup[1], cup[0], cup[3], cup[2]), fill=None, outline=(255,0,255))
draw_ellipse(pred_skeleton_img_edit1, (cup[1], cup[0], cup[3], cup[2]), outline=(255,0,255), width=r, antialias=8)
pred_skeleton_img_edit1.save('temp/skeleton_vessel_points_vis.png')
r = 1

# %%
