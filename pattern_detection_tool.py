import os
import cv2
import csv
import math
import glob
import random
import operator
import itertools
import scipy.misc
import numpy as np

from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from Roots_point_detect_tool import*

def show_pixel_set(img_nparray):
    a = img_nparray
    unique, counts = np.unique(a, return_counts=True)
    return dict(zip(unique, counts))

def tooth_forward(img_array):
    img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    half = int(img.shape[0]/2)
    img_up = img[:][int(half/10):half]
    img_down = img[:][half:int(half*19/10)]
    img_medium = img[:][int(half/2):int(half*3/2)]
    gums = 29
    bac = 0

    def get_counts(values, counts, traget_value):
        return counts[np.where(values == traget_value)] if counts[np.where(values == traget_value)].size > 0 else 1

    # Up
    values, counts = np.unique(img_up.astype(np.uint8), return_counts=True)
    sum_img_up_gums = get_counts(values, counts, gums)
    sum_img_up_bac = get_counts(values, counts, bac)

    # Down
    values, counts = np.unique(img_down.astype(np.uint8), return_counts=True)
    sum_img_down_gums = get_counts(values, counts, gums)
    sum_img_down_bac = get_counts(values, counts, bac)

    # Medium
    values, counts = np.unique(img_medium.astype(np.uint8), return_counts=True)
    sum_img_medium_gums = get_counts(values, counts, gums)
    sum_img_medium_bac = get_counts(values, counts, bac)

    up_rate = (sum_img_up_gums/sum_img_up_bac)
    down_rate = (sum_img_down_gums/sum_img_down_bac)

    if sum_img_medium_bac/sum_img_medium_gums > 3:
        forward, value = False, -1

    elif up_rate > down_rate and up_rate > 2.5:
        forward, value = True, 1
        # upper tooth
        
    elif up_rate < down_rate and down_rate > 2.5:
        forward, value = True, 0
        # lower tooth

    else:
        forward, value = False, -2

    return forward, value

def check_forward_status(value):
    if value == 1:
        print('This is upper tooth')
    elif value == 0:
        print('This is lower tooth')
    elif value == -1:
        print('This image have two raws of tooth')
    else:
        print('This is Unknow image')
        
def inertia_detection_cv_line(img_path, draw_img=None,nparray=True,scale=100,width=20,ignore_slope=False):
    if nparray:
        img = img_path
    else:
        img = scipy.misc.imread(img_path, flatten=1)
    x_v1, y_v1, x_mean, y_mean,len_non_zero = get_inertia_parameter(img)
    scale = scale
    try:
        try:
            if draw_img.all() != None:
                img = draw_img
        except:
            pass
        if len_non_zero > 5000 :
            if ignore_slope:
                cv2.line(img, (int(x_v1*-scale*2+x_mean),int(y_v1*-scale*2+y_mean)),(int(x_v1*scale*2+x_mean),int(y_v1*scale*2+y_mean)), (255, 0, 255), width)
                return img,[(int(x_v1*-scale*2+x_mean),int(y_v1*-scale*2+y_mean)),(int(x_v1*scale*2+x_mean),int(y_v1*scale*2+y_mean))]
            elif slope(-x_v1,-y_v1,x_v1,y_v1) > 1.2:
                cv2.line(img, (int(x_v1*-scale*2+x_mean),int(y_v1*-scale*2+y_mean)),(int(x_v1*scale*2+x_mean),int(y_v1*scale*2+y_mean)), (255, 0, 255), width)
                return img,[(int(x_v1*-scale*2+x_mean),int(y_v1*-scale*2+y_mean)),(int(x_v1*scale*2+x_mean),int(y_v1*scale*2+y_mean))]
            else:
                return img,[(0,0),(1,1)]
        else:
            return img,[(0,0),(1,1)]
    except Exception as e: 
        print('except')
        print(e)
        return img,[(0,0),(1,1)]
    
def slope(x1, y1, x2, y2):
    m = (y2-y1)/(x2-x1)
    return abs(m)

def draw_approx_hull_polygon(img):
    gray = img
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    black_img = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
    hulls = [cv2.convexHull(cnt) for cnt in contours]
    cv2.polylines(black_img, hulls, True, (255,0,0), 5)
    return black_img

def get_inertia_parameter(img_array):
    try:
        y, x = np.nonzero(img_array)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        x = x - x_mean
        y = y - y_mean
        coords = np.vstack([x, y])
        cov = np.cov(coords)
        evals, evecs = np.linalg.eig(cov)
        sort_indices = np.argsort(evals)[::-1]
        x_v1, y_v1 = evecs[:, sort_indices[0]]
        x_v2, y_v2 = evecs[:, sort_indices[1]]
        return x_v1, y_v1, x_mean, y_mean,len(x)
    except:
        return 4,3,2,1,0
    
def find_roots_by_countour(img_array1, img_array2, value):
    def find_all_pixels(image, color):
        return np.where(np.all(image == color, axis=2), 1, 0)

    red = np.array([255, 0, 0])
    pink = np.array([255, 0, 255])

    i, j = np.where(find_all_pixels(img_array1, red)*find_all_pixels(img_array2, pink) == 1)
    point_list = [[i_, j_] for i_, j_ in zip(i, j)]
    if point_list != []:
        if value == 1:
            return point_list[point_list.index(min(point_list))]
        elif value == 0:
            return point_list[point_list.index(max(point_list))]
    else:
        return [None, None]


def root_point_detect(fill_up_img,value):
    zero_img = np.zeros((fill_up_img.shape[0],fill_up_img.shape[1],3), np.uint8)
    inertia,line_of_inertia = inertia_detection_cv_line(fill_up_img,draw_img=zero_img,nparray=True,scale=500,width=3,ignore_slope=True)
    root_contour = draw_approx_hull_polygon(fill_up_img.astype('uint8'))
    y,x = find_roots_by_countour(root_contour,inertia,value)
    return [x],[y],line_of_inertia



def show_max_value_np(img_nparray):
    a = img_nparray
    unique, counts = np.unique(a, return_counts=True)
    dict_ = dict(zip(unique, counts))
    return max(dict_.items(), key=operator.itemgetter(1))[0]

def get_min_pink_pixel_point(point_list,img):
    count_dict = {get_count_pink(point,img):point for point in point_list}
    return count_dict[min(count_dict.keys())]

def get_count_pink(point,img):
    p =105
    x,y = point
    matrix = img[y-4:y+5,x-4:x+5]
    return sum(sum(np.where(matrix==p,1,0)))

def PBL_stage(value):
    if value == -999:
        return -999 
    elif value == 0:
        return 0
    elif value < 0.15:
        return 1
    elif value < 0.33:
        return 2
    else:
        return 3
    
def distance_point2line(point,line_edge1,line_edge2):
    return abs(np.cross(line_edge2-line_edge1,point-line_edge1)/np.linalg.norm(line_edge2-line_edge1))

def distance_2_points(point1,point2):
    return math.sqrt( ((point1[0]-point2[0])**2)+((point1[1]-point2[1])**2))

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return int(x), int(y)

def crosspoint_point2line(point,line_edge1,line_edge2):
    #hyparameter is three points in type np.array([x,y])
    m = -999
    try:
        m = (line_edge2[1]-line_edge1[1])/(line_edge2[0]-line_edge1[0])
        new_m = -1/m
        c = point[1]-new_m*point[0]
        point_2 = [point[0]+1,(point[0]+1)*new_m+c]
    except:
        if m == 0:
            point_2 = [point[0],point[1]+1]
        elif m == -999:
            point_2 = [point[0]+1,point[1]]

    return line_intersection((point,point_2), (line_edge1, line_edge2))

def point_in_left_side_of_line(point,line_edge1,line_edge2):
    dy = line_edge2[1]-line_edge1[1]
    dx = line_edge2[0]-line_edge1[0]
    if(dx==0):
        if(line_edge2[0]>point[0]):
            return True
        else:
            return False
    if(dy==0):
        dy = 1   
    try:
        m = (dy/dx)
    except:
        print('gradient error')
    c = line_edge1[1]-m*line_edge1[0]
    if (m < 0):
        if (point[1] - (m*point[0]) - c) < 0:
            return True
        else:
            return False
    if ((m > 0)):
        if (point[1] - (m*point[0]) - c) > 0:
            return True
        else:
            return False
    
    
    
# def find_cej_bone_keypoint(img_BGR_array):
#     kernel = np.ones((3,3),np.uint8)
#     img_gray = cv2.cvtColor(img_BGR_array,cv2.COLOR_BGR2GRAY)
#     tooth = cv2.morphologyEx(np.where(img_gray==94,1,0).astype('uint8'), cv2.MORPH_GRADIENT, kernel)
#     gums = cv2.morphologyEx(np.where(img_gray==29,1,0).astype('uint8'), cv2.MORPH_GRADIENT, kernel)
#     background = cv2.morphologyEx(np.where(img_gray==0,1,np.where(img_gray==86,1,np.where(img_gray==151,1,0))).astype('uint8'), cv2.MORPH_GRADIENT, kernel)
#     cej_crown = cv2.morphologyEx(np.where(img_gray==150,1,np.where(img_gray==179,1,0)).astype('uint8'), cv2.MORPH_GRADIENT, kernel)
#     depression = cv2.morphologyEx(np.where(img_gray==151,1,0).astype('uint8'), cv2.MORPH_GRADIENT, kernel)
    
#     cej = tooth*background*cej_crown
#     bone = tooth*background*gums
# #     cej = tooth*cej_crown
# #     bone = tooth*gums
    
#     cej_point = np.nonzero(cej)
#     bone_point = np.nonzero(bone)
#     return (cej_point[1],cej_point[0]), (bone_point[1],bone_point[0])
#     #return group_points_and_chose_best(cej_point[1],cej_point[0],img_gray),group_points_and_chose_best(bone_point[1],bone_point[0],img_gray)


def group_points_and_chose_best(list_x, list_y,img):
    new_list_x = []
    new_list_y = []  
    if list_x != [] and list_y != []:
        coords = [[x,y] for x, y in zip(list_x, list_y)]
        boolean = cdist(coords, coords) < 12
        matrix = [[] for __ in range(len(boolean))]
        for row_i, row in enumerate(boolean):
            matrix[row_i] = list([i for i, is_true in enumerate(row) if is_true])
        groups = []
        for pair in matrix:
            if pair not in groups and len(pair) > 1:
                groups.append(pair)
        for group in groups:
#             print(group)
            group_point = [coords[index] for index in group]
            group_point_array = np.array(group_point)
            x_cor = int((sum(group_point_array)/len(group_point_array))[0])
            y_cor = int((sum(group_point_array)/len(group_point_array))[1])
            if x_cor<img.shape[1] and y_cor< img.shape[0]:
                new_list_x.append(x_cor)
                new_list_y.append(y_cor)
    return new_list_x, new_list_y


# def find_cej_bone_keypoint(img_BGR_array):
#     kernel = np.ones((5,5),np.uint8)
#     img_gray = cv2.cvtColor(img_BGR_array,cv2.COLOR_BGR2GRAY)
    
#     tooth_area = np.where(img_gray==94,1,0).astype('uint8')
#     tooth_area_cv = cv2.morphologyEx(tooth_area, cv2.MORPH_OPEN, kernel)
#     tooth_area_cv = cv2.morphologyEx(tooth_area_cv, cv2.MORPH_CLOSE, kernel)
#     tooth_area_cv = cv2.dilate(tooth_area_cv,kernel,iterations = 1)
#     tooth_area_cv = tooth_area_cv*tooth_area
#     tooth = cv2.dilate(tooth_area_cv,kernel,iterations = 2)
# #     tooth = cv2.morphologyEx(tooth_area, cv2.MORPH_GRADIENT, kernel)
    
#     gums_area = np.where(img_gray==29,1,0).astype('uint8')
#     gums_area_cv = cv2.morphologyEx(gums_area, cv2.MORPH_OPEN, kernel)
#     gums_area_cv = cv2.morphologyEx(gums_area_cv, cv2.MORPH_CLOSE, kernel)
#     gums_area_cv = cv2.dilate(gums_area_cv,kernel,iterations = 1)
#     gums_area_cv = gums_area_cv*gums_area
#     gums = cv2.dilate(gums_area_cv,kernel,iterations = 2)
#     #     gums = cv2.morphologyEx(gums_area, cv2.MORPH_GRADIENT, kernel)
    
#     background_area = np.where(img_gray==0,1,np.where(img_gray==86,1,np.where(img_gray==151,1,0))).astype('uint8')
#     background_area_cv = cv2.morphologyEx(background_area, cv2.MORPH_OPEN, kernel)
#     background_area_cv = cv2.morphologyEx(background_area_cv, cv2.MORPH_CLOSE, kernel)
#     background_area_cv = cv2.dilate(background_area_cv,kernel,iterations = 1)
#     background_area_cv = background_area_cv*background_area
#     background = cv2.dilate(background_area_cv,kernel,iterations = 2)
#     #background = cv2.morphologyEx(background_area, cv2.MORPH_GRADIENT, kernel)
    
#     cej_crown_area = np.where(img_gray==150,1,np.where(img_gray==179,1,0)).astype('uint8')
#     cej_crown_area_cv = cv2.morphologyEx(cej_crown_area, cv2.MORPH_OPEN, kernel)
#     cej_crown_area_cv = cv2.morphologyEx(cej_crown_area_cv, cv2.MORPH_CLOSE, kernel)
#     cej_crown_area_cv = cv2.dilate(cej_crown_area_cv,kernel,iterations = 1)
#     cej_crown_area_cv = cej_crown_area_cv*cej_crown_area
#     cej_crown = cv2.dilate(cej_crown_area_cv,kernel,iterations = 2)
#     #cej_crown = cv2.morphologyEx(cej_crown_area_cv, cv2.MORPH_GRADIENT, kernel)
    

#     cej = tooth*background*cej_crown*tooth_area
#     bone = tooth*background*gums*gums_area
#     cej_point = np.nonzero(cej)
#     bone_point = np.nonzero(bone)

#     return group_points_and_chose_best(cej_point[1],cej_point[0],img_gray),group_points_and_chose_best(bone_point[1],bone_point[0],img_gray)


def cej_recover(cej_x_list, cej_y_list, line_of_inertia, fill_up_img):
    blank_image = np.zeros((fill_up_img.shape[0],fill_up_img.shape[1]), np.uint8)
    cej_x = cej_x_list[0]
    cej_y = cej_y_list[0]
    cej_cross_point_with_inertia = crosspoint_point2line((cej_x, cej_y),line_of_inertia[0],line_of_inertia[1])
    center_x = cej_cross_point_with_inertia[0]
    center_y = cej_cross_point_with_inertia[1]
    vecter_x = (center_x-cej_x)*50
    vecter_y = (center_y-cej_y)*50
    cv2.line(blank_image, (cej_x-vecter_x, cej_y-vecter_y), (center_x+vecter_x, center_y+vecter_y), (1), 5)
#     plt.imshow(blank_image)
#     plt.show()
    target_line = blank_image*(fill_up_img/255)
#     plt.imshow(target_line)
#     plt.show()
    (y_, x_) = np.nonzero(target_line)
    point_list = [(y, x) for y ,x in zip(y_,x_)]
    point_list.sort(key = lambda x: x[1])
#     plt.imshow(fill_up_img)
#     plt.scatter(x=cej_x, y=cej_y, c='r', s=8)
#     plt.scatter(x=point_list[0][1], y=point_list[0][0], c='g', s=8)
#     plt.scatter(x=point_list[-1][1], y=point_list[-1][0], c='g', s=8)
#     plt.show()
    if len(point_list) > 1:
        return ([point_list[0][1],point_list[-1][1]], [point_list[0][0],point_list[-1][0]])
    else:
        return ([cej_x], [cej_y])

    
# def bone_recover(bone_x_list, bone_y_list, line_of_inertia, seg_img_with_cej_img, value):
#     kernel = np.ones((5,5),np.uint8)
#     img_gray = cv2.cvtColor(seg_img_with_cej_img,cv2.COLOR_BGR2GRAY)
#     tooth = cv2.morphologyEx(np.where(img_gray==94,1,0).astype('uint8'), cv2.MORPH_GRADIENT, kernel)
#     gums = cv2.morphologyEx(np.where(img_gray==29,1,0).astype('uint8'), cv2.MORPH_GRADIENT, kernel)
#     background = cv2.morphologyEx(np.where(img_gray==0,1,np.where(img_gray==86,1,np.where(img_gray==151,1,0))).astype('uint8'), cv2.MORPH_GRADIENT, kernel)
    
#     if bone_x_list !=[]:

#         bone_x = bone_x_list[0]
#         bone_y = bone_y_list[0]
#         point_state = point_in_left_side_of_line((bone_x, bone_y), line_of_inertia[0], line_of_inertia[1])
#         point_target_state = not point_state

#         (gums_y_, gums_x_) = np.nonzero(tooth*gums)
#         gum_point_list = [(y, x) for y ,x in zip(gums_y_,gums_x_) if point_in_left_side_of_line((x,y), line_of_inertia[0],line_of_inertia[1]) == point_target_state]
#         gum_point_list.sort(key = lambda x: x[0])


#         if value == 0:
#             try:
#                 gum_hight = gum_point_list[0][0]
#             except:
#                 gum_hight = seg_img_with_cej_img.shape[0]
#         elif value == 1:
#             try:
#                 gum_hight = gum_point_list[-1][0]
#             except:
#                 gum_hight = 0
                
#         (bac_y_, bac_x_) = np.nonzero(tooth*background)
#         bac_point_list = [(y, x) for y ,x in zip(bac_y_, bac_x_) if point_in_left_side_of_line((x,y), line_of_inertia[0],line_of_inertia[1]) == point_target_state]
#         if value == 0:
#             bac_point_list_filter = [point for point in bac_point_list if point[0] < gum_hight]
#         elif value == 1:
#             bac_point_list_filter = [point for point in bac_point_list if point[0] > gum_hight]
#         bac_point_list_filter.sort(key = lambda x: x[0])

#         if len(bac_point_list_filter)>1:
#             if value == 0:
#                 return  ([bone_x, bac_point_list_filter[-1][1]], [bone_y, bac_point_list_filter[-1][0]])
#             elif value == 1:
#                 return  ([bone_x, bac_point_list_filter[0][1]], [bone_y, bac_point_list_filter[0][0]])
#         else:
#              return  ([bone_x], [bone_y])
#     else:
#         (gums_y_, gums_x_) = np.nonzero(tooth*gums)
        
#         gum_point_list_left = [(y, x) for y, x in zip(gums_y_,gums_x_) if point_in_left_side_of_line((x,y), line_of_inertia[0],line_of_inertia[1]) == True]
#         gum_point_list_right = [(y, x) for y, x in zip(gums_y_,gums_x_) if point_in_left_side_of_line((x,y), line_of_inertia[0],line_of_inertia[1]) == False]
#         gum_point_list_left.sort(key = lambda x: x[0])
#         gum_point_list_right.sort(key = lambda x: x[0])
        
# #         if value == 0:
# #             gum_hight_left = gum_point_list_left[0][0]
# #             gum_hight_right = gum_point_list_right[0][0]
# #         elif value == 1:
# #             gum_hight_left = gum_point_list_left[-1][0]
# #             gum_hight_right = gum_point_list_right[-1][0]
            
#         if value == 0:
#             try:
#                 gum_hight_left = gum_point_list_left[0][0]
#             except:
#                 gum_hight_left = seg_img_with_cej_img.shape[0]
#             try:
#                 gum_hight_right = gum_point_list_right[0][0]
#             except:
#                 gum_hight_right = seg_img_with_cej_img.shape[0]
#         elif value == 1:
#             try:
#                 gum_hight_left = gum_point_list_left[-1][0]
#             except:
#                 gum_hight_left = 0
#             try:
#                 gum_hight_right = gum_point_list_right[-1][0]
#             except:
#                 gum_hight_right = 0
                
#         (bac_y_, bac_x_) = np.nonzero(tooth*background)

#         bac_point_list_left = [(y, x) for y ,x in zip(bac_y_, bac_x_) if point_in_left_side_of_line((x,y), line_of_inertia[0],line_of_inertia[1]) == True]
#         bac_point_list_right = [(y, x) for y ,x in zip(bac_y_, bac_x_) if point_in_left_side_of_line((x,y), line_of_inertia[0],line_of_inertia[1]) == False]
        
#         if value == 0:
#             bac_point_list_left_filter = [point for point in bac_point_list_left if point[0] < gum_hight_left]
#             bac_point_list_right_filter = [point for point in bac_point_list_right if point[0] < gum_hight_right]
#         elif value == 1:
#             bac_point_list_left_filter = [point for point in bac_point_list_left if point[0] > gum_hight_left]
#             bac_point_list_right_filter = [point for point in bac_point_list_right if point[0] > gum_hight_right]

#         bac_point_list_left_filter.sort(key = lambda x: x[0])
#         bac_point_list_right_filter.sort(key = lambda x: x[0])

#         bone_x_new = []
#         bone_y_new = []
#         if len(bac_point_list_left_filter)>0:
#             if value == 0:
#                 bone_x_new.append(bac_point_list_left_filter[-1][1])
#                 bone_y_new.append(bac_point_list_left_filter[-1][0])
#             elif value == 1:
#                 bone_x_new.append(bac_point_list_left_filter[0][1])
#                 bone_y_new.append(bac_point_list_left_filter[0][0])
            
#         if len(bac_point_list_right_filter)>0:
#             if value == 0:
#                 bone_x_new.append(bac_point_list_right_filter[-1][1])
#                 bone_y_new.append(bac_point_list_right_filter[-1][0])
#             elif value == 1:
#                 bone_x_new.append(bac_point_list_right_filter[0][1])
#                 bone_y_new.append(bac_point_list_right_filter[0][0])
                
#         return  (bone_x_new, bone_y_new)

# def bone_recover(bone_x_list, bone_y_list, fill_up_img, line_of_inertia, seg_img_with_cej_img, value):

#     kernel = np.ones((5,5),np.uint8)
#     img_gray = cv2.cvtColor(seg_img_with_cej_img,cv2.COLOR_BGR2GRAY)
#     tooth = cv2.morphologyEx(np.where(img_gray==94,1,0).astype('uint8'), cv2.MORPH_GRADIENT, kernel)
#     gums = cv2.morphologyEx(np.where(img_gray==29,1,0).astype('uint8'), cv2.MORPH_GRADIENT, kernel)
    
#     background_mask = np.where(img_gray==0,1,np.where(img_gray==86,1,np.where(img_gray==151,1,0))).astype('uint8')
#     background_mask_rm_p = remove_pieces_50(background_mask)
#     background = cv2.morphologyEx(background_mask_rm_p, cv2.MORPH_GRADIENT, kernel)
    
#     if bone_x_list !=[]:
#         bone_x = bone_x_list[0]
#         bone_y = bone_y_list[0]
#         point_state = point_in_left_side_of_line((bone_x, bone_y), line_of_inertia[0], line_of_inertia[1])
#         point_target_state = not point_state

#         (gums_y_, gums_x_) = np.nonzero(tooth*gums)
#         gum_point_list = [(y, x) for y ,x in zip(gums_y_,gums_x_) if point_in_left_side_of_line((x,y), line_of_inertia[0],line_of_inertia[1]) == point_target_state]
#         gum_point_list.sort(key = lambda x: x[0])

#         if value == 0:
#             try:
#                 gum_hight = gum_point_list[0][0]
#             except:
#                 gum_hight = seg_img_with_cej_img.shape[0]
#         elif value == 1:
#             try:
#                 gum_hight = gum_point_list[-1][0]
#             except:
#                 gum_hight = 0
#         (bac_y_, bac_x_) = np.nonzero(tooth*background)
#         bac_point_list = [(y, x) for y ,x in zip(bac_y_, bac_x_) if point_in_left_side_of_line((x,y), line_of_inertia[0],line_of_inertia[1]) == point_target_state]
#         if value == 0:
#             bac_point_list_filter = [point for point in bac_point_list if point[0] < gum_hight]
#         elif value == 1:
#             bac_point_list_filter = [point for point in bac_point_list if point[0] > gum_hight]
#         bac_point_list_filter.sort(key = lambda x: x[0])

        
#         if len(bac_point_list_filter)>1:
#             if value == 0:
#                 #(x, y)
#                 Candidate_point = (bac_point_list_filter[-1][1], bac_point_list_filter[-1][0])
#                 #(x, y)
#                 Candidate_cross_point_with_inertia = crosspoint_point2line(Candidate_point ,line_of_inertia[0],line_of_inertia[1])
#                 #(y, x)
#                 Candidate_cross_point_with_inertia = Candidate_cross_point_with_inertia[::-1]
#                 bone_point_distance_list = [distance_2_points(Candidate_cross_point_with_inertia, point) for point in bac_point_list_filter]
#                 target_point_index = bone_point_distance_list.index(min(bone_point_distance_list))
#                 return  ([bone_x, bac_point_list_filter[target_point_index][1]], [bone_y, bac_point_list_filter[target_point_index][0]])
#             elif value == 1:
#                 #(x, y)
#                 Candidate_point = (bac_point_list_filter[0][1], bac_point_list_filter[0][0])
#                 #(x, y)
#                 Candidate_cross_point_with_inertia = crosspoint_point2line(Candidate_point ,line_of_inertia[0],line_of_inertia[1])
#                 #(x, y) -> (y, x)
#                 Candidate_cross_point_with_inertia = Candidate_cross_point_with_inertia[::-1]
#                 bone_point_distance_list = [distance_2_points(Candidate_cross_point_with_inertia, point) for point in bac_point_list_filter]
#                 target_point_index = bone_point_distance_list.index(min(bone_point_distance_list))
#                 return  ([bone_x, bac_point_list_filter[target_point_index][1]], [bone_y, bac_point_list_filter[target_point_index][0]])
#         else:
#              return  ([bone_x], [bone_y])
#     else:

#         (gums_y_, gums_x_) = np.nonzero(tooth*gums)
#         gum_point_list_left = [(y, x) for y, x in zip(gums_y_,gums_x_) if point_in_left_side_of_line((x,y), line_of_inertia[0],line_of_inertia[1]) == True]
#         gum_point_list_right = [(y, x) for y, x in zip(gums_y_,gums_x_) if point_in_left_side_of_line((x,y), line_of_inertia[0],line_of_inertia[1]) == False]
#         gum_point_list_left.sort(key = lambda x: x[0])
#         gum_point_list_right.sort(key = lambda x: x[0])
            
#         if value == 0:
#             try:
#                 gum_hight_left = gum_point_list_left[0][0]+5
#             except:
#                 gum_hight_left = seg_img_with_cej_img.shape[0]
#             try:
#                 gum_hight_right = gum_point_list_right[0][0]+5
#             except:
#                 gum_hight_right = seg_img_with_cej_img.shape[0]
#         elif value == 1:
#             try:
#                 gum_hight_left = gum_point_list_left[-1][0]-5
#             except:
#                 gum_hight_left = 0
#             try:
#                 gum_hight_right = gum_point_list_right[-1][0]-5
#             except:
#                 gum_hight_right = 0

#         (bac_y_, bac_x_) = np.nonzero(tooth*background)

#         bac_point_list_left = [(y, x) for y ,x in zip(bac_y_, bac_x_) if point_in_left_side_of_line((x,y), line_of_inertia[0],line_of_inertia[1]) == True]
#         bac_point_list_right = [(y, x) for y ,x in zip(bac_y_, bac_x_) if point_in_left_side_of_line((x,y), line_of_inertia[0],line_of_inertia[1]) == False]
        
#         if value == 0:
#             bac_point_list_left_filter = [point for point in bac_point_list_left if point[0] < gum_hight_left]
#             bac_point_list_right_filter = [point for point in bac_point_list_right if point[0] < gum_hight_right]
#         elif value == 1:
#             bac_point_list_left_filter = [point for point in bac_point_list_left if point[0] > gum_hight_left]
#             bac_point_list_right_filter = [point for point in bac_point_list_right if point[0] > gum_hight_right]

#         bac_point_list_left_filter.sort(key = lambda x: x[0])
#         bac_point_list_right_filter.sort(key = lambda x: x[0])

#         bone_x_new = []
#         bone_y_new = []
#         if len(bac_point_list_left_filter)>0:
#             if value == 0:
#                 #(x, y)
#                 Candidate_point = (bac_point_list_left_filter[-1][1], bac_point_list_left_filter[-1][0]+40)
#                 #(x, y)
#                 Candidate_cross_point_with_inertia = crosspoint_point2line(Candidate_point ,line_of_inertia[0],line_of_inertia[1])
#                 #(y, x)
#                 Candidate_cross_point_with_inertia = Candidate_cross_point_with_inertia[::-1]
#                 bone_point_distance_list = [distance_2_points(Candidate_cross_point_with_inertia, point) for point in bac_point_list_left_filter]
#                 target_point_index = bone_point_distance_list.index(min(bone_point_distance_list))
#                 bone_x_new.append(bac_point_list_left_filter[target_point_index][1])
#                 bone_y_new.append(bac_point_list_left_filter[target_point_index][0])
                
# #                 bone_x_new.append(bac_point_list_left_filter[-1][1])
# #                 bone_y_new.append(bac_point_list_left_filter[-1][0])
#             elif value == 1:
#                 #(x, y)
#                 Candidate_point = (bac_point_list_left_filter[0][1], bac_point_list_left_filter[0][0]-40)
#                 #(x, y)
#                 Candidate_cross_point_with_inertia = crosspoint_point2line(Candidate_point ,line_of_inertia[0],line_of_inertia[1])
#                 #(x, y) -> (y, x)
#                 Candidate_cross_point_with_inertia = Candidate_cross_point_with_inertia[::-1]
#                 bone_point_distance_list = [distance_2_points(Candidate_cross_point_with_inertia, point) for point in bac_point_list_left_filter]
#                 target_point_index = bone_point_distance_list.index(min(bone_point_distance_list))

#                 bone_x_new.append(bac_point_list_left_filter[target_point_index][1])
#                 bone_y_new.append(bac_point_list_left_filter[target_point_index][0])
            
#         if len(bac_point_list_right_filter)>0:
#             if value == 0:
#                 #(x, y)
#                 Candidate_point = (bac_point_list_right_filter[-1][1], bac_point_list_right_filter[-1][0]+40)
#                 #(x, y)
#                 Candidate_cross_point_with_inertia = crosspoint_point2line(Candidate_point ,line_of_inertia[0],line_of_inertia[1])
#                 #(y, x)
#                 Candidate_cross_point_with_inertia = Candidate_cross_point_with_inertia[::-1]
#                 bone_point_distance_list = [distance_2_points(Candidate_cross_point_with_inertia, point) for point in bac_point_list_right_filter]
#                 target_point_index = bone_point_distance_list.index(min(bone_point_distance_list))
                
#                 bone_x_new.append(bac_point_list_right_filter[target_point_index][1])
#                 bone_y_new.append(bac_point_list_right_filter[target_point_index][0])   

#             elif value == 1:
#                 #(x, y)
#                 Candidate_point = (bac_point_list_right_filter[0][1], bac_point_list_right_filter[0][0]-40)
#                 #(x, y)
#                 Candidate_cross_point_with_inertia = crosspoint_point2line(Candidate_point ,line_of_inertia[0],line_of_inertia[1])
#                 #(x, y) -> (y, x)
#                 Candidate_cross_point_with_inertia = Candidate_cross_point_with_inertia[::-1]
#                 bone_point_distance_list = [distance_2_points(Candidate_cross_point_with_inertia, point) for point in bac_point_list_right_filter]
#                 target_point_index = bone_point_distance_list.index(min(bone_point_distance_list))
#                 bone_x_new.append(bac_point_list_right_filter[target_point_index][1])
#                 bone_y_new.append(bac_point_list_right_filter[target_point_index][0])  
                
#         return  (bone_x_new, bone_y_new)
    
    
# def get_cej_point_from_contour(cej_contour,fill_up_img , line_of_inertia, value, molar):
#     (cej_points_y,cej_points_x) = np.nonzero(cej_contour)
#     cej_points = [(x, y) for x, y in zip(cej_points_x, cej_points_y)]
#     root_x, root_y, line_of_inertia_L = get_molar_root(fill_up_img, value, line_of_inertia, 'L')
#     cej_point_list_L = [cej for cej in cej_points if point_in_left_side_of_line(cej, line_of_inertia_L[0], line_of_inertia_L[1])]
    
#     root_x, root_y, line_of_inertia_R = get_molar_root(fill_up_img, value, line_of_inertia, 'R')
#     cej_point_list_R = [cej for cej in cej_points if not point_in_left_side_of_line(cej, line_of_inertia_R[0], line_of_inertia_R[1])]
#     if value == 0:
#         reverse_ = False
#     elif value == 1:
#         reverse_ = True
#     cej_point_list_L.sort(key = lambda x:x[1], reverse = reverse_)
#     cej_point_list_R.sort(key = lambda x:x[1], reverse = reverse_)
#     cej_x = []
#     cej_y = []
#     try:
#         cej_x.append(cej_point_list_L[0][0])
#         cej_y.append(cej_point_list_L[0][1])
#     except:
#         pass
#     try:
#         cej_x.append(cej_point_list_R[0][0])
#         cej_y.append(cej_point_list_R[0][1])
#     except:
#         pass
#     return cej_x, cej_y

# def get_bone_point_from_contour(bone_contour,fill_up_img, line_of_inertia, value, molar):
#     (bone_points_y, bone_points_x) = np.nonzero(bone_contour)
#     bone_points = [(x, y) for x, y in zip(bone_points_x, bone_points_y)]
    
#     root_x, root_y, line_of_inertia_L = get_molar_root(fill_up_img, value, line_of_inertia, 'L')
#     bone_point_list_L = [bone for bone in bone_points if point_in_left_side_of_line(bone, line_of_inertia_L[0], line_of_inertia_L[1])]
    
#     root_x, root_y, line_of_inertia_R = get_molar_root(fill_up_img, value, line_of_inertia, 'R')
#     bone_point_list_R = [bone for bone in bone_points if not point_in_left_side_of_line(bone, line_of_inertia_R[0], line_of_inertia_R[1])]
#     if value == 0:
#         reverse_ = True
#     elif value == 1:
#         reverse_ = False
#     bone_point_list_L.sort(key = lambda x:x[1], reverse = reverse_)
#     bone_point_list_R.sort(key = lambda x:x[1], reverse = reverse_)
#     bone_x = []
#     bone_y = []
#     try:
#         bone_x.append(bone_point_list_L[0][0])
#         bone_y.append(bone_point_list_L[0][1])
#     except:
#         pass
#     try:
#         bone_x.append(bone_point_list_R[0][0])
#         bone_y.append(bone_point_list_R[0][1])
#     except:
#         pass
#     return bone_x, bone_y

def get_molar_root(fill_up_img, value, line_of_inertia, type_):
    y_list,x_list = np.nonzero(fill_up_img)
    
    L_fill_up = fill_up_img.copy()
    if type_ == 'L':
        point_in_left = [point_in_left_side_of_line([x_,y_],line_of_inertia[0],line_of_inertia[1]) for (x_, y_) in zip(x_list,y_list)]
        for bool_, x_, y_ in zip(point_in_left, x_list, y_list):
            if (not bool_):
                L_fill_up[y_,x_] = 0
        root_x, root_y,line_of_inertia = root_point_detect(L_fill_up,value)
#         plt.imshow(L_fill_up)
#         plt.show()
        return root_x, root_y,line_of_inertia

    if type_ == 'R':
        R_fill_up = fill_up_img.copy()
        point_in_left = [point_in_left_side_of_line([x_,y_],line_of_inertia[0],line_of_inertia[1]) for (x_, y_) in zip(x_list,y_list)]
        for bool_, x_, y_ in zip(point_in_left, x_list, y_list):
            if bool_:
                R_fill_up[y_,x_] = 0          
        root_x, root_y,line_of_inertia = root_point_detect(R_fill_up,value)
#         plt.imshow(R_fill_up)
#         plt.show()
        return root_x, root_y,line_of_inertia


def PolygonArea(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

def point_in_box(point_boxs_list,check_point):
    point_pair_list = [[point_boxs_list[i-1],point_boxs_list[i]] for i in range(len(point_boxs_list))]
    box_area = PolygonArea(point_boxs_list)
    count_area = 0
    for edge in point_pair_list:
        edge.append(check_point)
        count_area = PolygonArea(edge)+count_area
    if count_area - box_area < 1:
        return True
    else:
        return False
    
def get_molar_bool(json_file_path, check_point):
    with open(json_file_path, 'r') as file:
        json_file = json.load(file)
    for tooth in json_file['shapes']:
        check_in_box = point_in_box(tooth['points'],check_point)
        if check_in_box:
            tooth_num = int(tooth['label'].split('_')[0])
            if (tooth_num >= 1 and tooth_num <=3) or (tooth_num >= 14 and tooth_num <=19) or (tooth_num >= 30 and tooth_num <=32):
                return True
            else:
                return False
    return False

#bone is follow to cej three area
def keypoint_filter(cej_points, bone_points, line_of_inertia, value, fill_up_img):
    new_list_x = []
    new_list_y = []
    cej_points = list(sorted(cej_points, key = lambda x:x[0]))
    bone_points = list(sorted(bone_points, key = lambda x:x[0]))

    root_x, root_y, line_of_inertia_L = get_molar_root(fill_up_img, value, line_of_inertia, 'L')
    cej_num_list_L = [point_in_left_side_of_line(cej,line_of_inertia_L[0],line_of_inertia_L[1]) for cej in cej_points]
    root_x, root_y, line_of_inertia_R = get_molar_root(fill_up_img, value, line_of_inertia, 'R')
    cej_num_list_R = [point_in_left_side_of_line(cej,line_of_inertia_R[0],line_of_inertia_R[1]) for cej in cej_points]
    
    cej_left_points = [cej for bool_left,cej in zip(cej_num_list_L,cej_points) if bool_left]
    cej_right_points = [cej for bool_left,cej in zip(cej_num_list_R,cej_points) if not bool_left]
    
    cej_x = []
    cej_y = []
    #upper tooth
    if (value==1):
        try:
            cej_left_points = list(sorted(cej_left_points, key = lambda x:x[0], reverse=True))[0]
            cej_points.append(cej_left_points)
            cej_x.append(cej_left_points[0])
            cej_y.append(cej_left_points[1])
        except:
            pass
        try:
            cej_right_points = list(sorted(cej_right_points, key = lambda x:x[0]))[0]
            cej_points.append(cej_right_points)
            cej_x.append(cej_right_points[0])
            cej_y.append(cej_right_points[1])
        except:
            pass
    #lower tooth
    if (value==0):
        try:
            cej_left_points = list(sorted(cej_left_points, key = lambda x:x[0], reverse=True))[0]
            cej_points.append(cej_left_points)
            cej_x.append(cej_left_points[0])
            cej_y.append(cej_left_points[1])
        except:
            pass
        try:
            cej_right_points = list(sorted(cej_right_points, key = lambda x:x[0]))[0]
            cej_points.append(cej_right_points)
            cej_x.append(cej_right_points[0])
            cej_y.append(cej_right_points[1])
        except:
            pass
    
    cej_final_points = [(x,y) for x,y in zip(cej_x,cej_y)]
    cej_left_bool_list = [point_in_left_side_of_line(cej,line_of_inertia[0],line_of_inertia[1]) for cej in cej_final_points]
    
    root_x, root_y, line_of_inertia_L = get_molar_root(fill_up_img, value, line_of_inertia, 'L')
    bone_num_list_L = [point_in_left_side_of_line(bone, line_of_inertia_L[0], line_of_inertia_L[1]) for bone in bone_points]
    root_x, root_y, line_of_inertia_R = get_molar_root(fill_up_img, value, line_of_inertia, 'R')
    bone_num_list_R = [point_in_left_side_of_line(bone, line_of_inertia_R[0], line_of_inertia_R[1]) for bone in bone_points]
 
    bone_left_points = [bone for bool_left,bone in zip(bone_num_list_L,bone_points) if  bool_left]
    bone_right_points = [bone for bool_left,bone in zip(bone_num_list_R,bone_points) if not bool_left]
    

    bone_x = []
    bone_y = []

    
    #upper tooth
    if (value==1):
        try:
            bone_left_points = list(sorted(bone_left_points, key = lambda x:x[1],reverse=True))
            bone_left_points_exclude_on_crown = [(x,y) for (x,y) in bone_left_points if y < cej_y[cej_left_bool_list.index(True)]]
            best_point = bone_left_points_exclude_on_crown[0]
            bone_x.append(best_point[0])
            bone_y.append(best_point[1])
        except:
            pass
        try:
            bone_right_points = list(sorted(bone_right_points, key = lambda x:x[1],reverse=True))
            bone_right_points_exclude_on_crown = [(x,y) for (x,y) in bone_right_points if y < cej_y[cej_left_bool_list.index(False)]]
            best_point = bone_right_points_exclude_on_crown[0]
            bone_x.append(best_point[0])
            bone_y.append(best_point[1])
        except:
            pass
    #lower tooth
    if (value==0):
        try:
            bone_left_points = list(sorted(bone_left_points, key = lambda x:x[1],reverse=False))
            bone_left_points_exclude_on_crown = [(x,y) for (x,y) in bone_left_points if y > cej_y[cej_left_bool_list.index(True)]]
            best_point = bone_left_points_exclude_on_crown[0]
            bone_x.append(best_point[0])
            bone_y.append(best_point[1])
        except:
            pass
        try:
            bone_right_points = list(sorted(bone_right_points, key = lambda x:x[1],reverse=False))
            bone_right_points_exclude_on_crown = [(x,y) for (x,y) in bone_right_points if y > cej_y[cej_left_bool_list.index(False)]]
            best_point = bone_right_points_exclude_on_crown[0]
            bone_x.append(best_point[0])
            bone_y.append(best_point[1])
        except:
            pass
#     print("cej_x:",cej_x,"cej_y:",cej_y)
#     print("bone_x:",bone_x,"bone_y:",bone_y)
#     print("-------------------------------")
    return (cej_x,cej_y),(bone_x,bone_y)

# #bone is follow to cej three area
# def keypoint_filter(cej_points, bone_points, line_of_inertia, value, fill_up_img):
#     new_list_x = []
#     new_list_y = []
#     cej_points = list(sorted(cej_points, key = lambda x:x[0]))
#     bone_points = list(sorted(bone_points, key = lambda x:x[0]))
    
# #     root_x, root_y, line_of_inertia_L = get_molar_root(fill_up_img, value, line_of_inertia, 'L')
# #     root_x, root_y, line_of_inertia_R = get_molar_root(fill_up_img, value, line_of_inertia, 'R')
    
# #     cej_num_list = [point_in_left_side_of_line(cej,line_of_inertia[0],line_of_inertia[1]) for cej in cej_points]
# #     bone_num_list = [point_in_left_side_of_line(bone,line_of_inertia[0],line_of_inertia[1]) for bone in bone_points]

#     root_x, root_y, line_of_inertia_L = get_molar_root(fill_up_img, value, line_of_inertia, 'L')
#     cej_num_list_L = [point_in_left_side_of_line(cej,line_of_inertia_L[0],line_of_inertia_L[1]) for cej in cej_points]
#     root_x, root_y, line_of_inertia_R = get_molar_root(fill_up_img, value, line_of_inertia, 'R')
#     cej_num_list_R = [point_in_left_side_of_line(cej,line_of_inertia_R[0],line_of_inertia_R[1]) for cej in cej_points]
    
# #     bone_num_list = [point_in_left_side_of_line(bone,line_of_inertia[0],line_of_inertia[1]) for bone in bone_points]
    
    
# #     cej_left_points = [cej for bool_left,cej in zip(cej_num_list,cej_points) if bool_left]
# #     cej_right_points = [cej for bool_left,cej in zip(cej_num_list,cej_points) if not bool_left]
#     cej_left_points = [cej for bool_left,cej in zip(cej_num_list_L,cej_points) if bool_left]
#     cej_right_points = [cej for bool_left,cej in zip(cej_num_list_R,cej_points) if not bool_left]
    
#     cej_x = []
#     cej_y = []
#     #upper tooth
#     if (value==1):
#         try:
#             cej_left_points = list(sorted(cej_left_points, key = lambda x:x[1]))[0]
#             cej_points.append(cej_left_points)
#             cej_x.append(cej_left_points[0])
#             cej_y.append(cej_left_points[1])
#         except:
#             pass
#         try:
#             cej_right_points = list(sorted(cej_right_points, key = lambda x:x[1]))[0]
#             cej_points.append(cej_right_points)
#             cej_x.append(cej_right_points[0])
#             cej_y.append(cej_right_points[1])
#         except:
#             pass
#     #lower tooth
#     if (value==0):
#         try:
#             cej_left_points = list(sorted(cej_left_points, key = lambda x:x[1],reverse=True))[0]
#             cej_points.append(cej_left_points)
#             cej_x.append(cej_left_points[0])
#             cej_y.append(cej_left_points[1])
#         except:
#             pass
#         try:
#             cej_right_points = list(sorted(cej_right_points, key = lambda x:x[1],reverse=True))[0]
#             cej_points.append(cej_right_points)
#             cej_x.append(cej_right_points[0])
#             cej_y.append(cej_right_points[1])
#         except:
#             pass
    
#     cej_final_points = [(x,y) for x,y in zip(cej_x,cej_y)]
#     cej_left_bool_list = [point_in_left_side_of_line(cej,line_of_inertia[0],line_of_inertia[1]) for cej in cej_final_points]
    
#     root_x, root_y, line_of_inertia_L = get_molar_root(fill_up_img, value, line_of_inertia, 'L')
#     bone_num_list_L = [point_in_left_side_of_line(bone, line_of_inertia_L[0], line_of_inertia_L[1]) for bone in bone_points]
#     root_x, root_y, line_of_inertia_R = get_molar_root(fill_up_img, value, line_of_inertia, 'R')
#     bone_num_list_R = [point_in_left_side_of_line(bone, line_of_inertia_R[0], line_of_inertia_R[1]) for bone in bone_points]
 
#     bone_left_points = [bone for bool_left,bone in zip(bone_num_list_L,bone_points) if  bool_left]
#     bone_right_points = [bone for bool_left,bone in zip(bone_num_list_R,bone_points) if not bool_left]
    
# #     bone_left_points = [bone for bool_left,bone in zip(bone_num_list,bone_points) if  bool_left]
# #     bone_right_points = [bone for bool_left,bone in zip(bone_num_list,bone_points) if not bool_left]
    
#     bone_x = []
#     bone_y = []
# #     print("-------------------------------")
# #     print("line_of_inertia[0],line_of_inertia[1]",line_of_inertia[0],line_of_inertia[1])
# #     print("cej_final_points:",cej_final_points)
# #     print("cej_final_side:",cej_num_list)
    
# #     print(len(bone_num_list))
# #     print("cej_num_list:",cej_num_list)
# #     print("cej_right_points:",cej_right_points)
# #     print("cej_left_points:",cej_left_points)
    
# #     print("bone_num_list:",bone_num_list)
# #     print("bone_right_points:",bone_right_points)
# #     print("bone_left_points:",bone_left_points)
# #     print("cej_x:",cej_x)
# #     print("cej_y:",cej_y)
    
#     #upper tooth
#     if (value==1):
#         try:
#             bone_left_points = list(sorted(bone_left_points, key = lambda x:x[1],reverse=True))
#             bone_left_points_exclude_on_crown = [(x,y) for (x,y) in bone_left_points if y < cej_y[cej_left_bool_list.index(True)]]
#             best_point = bone_left_points_exclude_on_crown[0]
#             bone_x.append(best_point[0])
#             bone_y.append(best_point[1])
#         except:
#             pass
#         try:
#             bone_right_points = list(sorted(bone_right_points, key = lambda x:x[1],reverse=True))
#             bone_right_points_exclude_on_crown = [(x,y) for (x,y) in bone_right_points if y < cej_y[cej_left_bool_list.index(False)]]
#             best_point = bone_right_points_exclude_on_crown[0]
#             bone_x.append(best_point[0])
#             bone_y.append(best_point[1])
#         except:
#             pass
#     #lower tooth
#     if (value==0):
#         try:
#             bone_left_points = list(sorted(bone_left_points, key = lambda x:x[1],reverse=False))
#             bone_left_points_exclude_on_crown = [(x,y) for (x,y) in bone_left_points if y > cej_y[cej_left_bool_list.index(True)]]
#             best_point = bone_left_points_exclude_on_crown[0]
#             bone_x.append(best_point[0])
#             bone_y.append(best_point[1])
#         except:
#             pass
#         try:
#             bone_right_points = list(sorted(bone_right_points, key = lambda x:x[1],reverse=False))
#             bone_right_points_exclude_on_crown = [(x,y) for (x,y) in bone_right_points if y > cej_y[cej_left_bool_list.index(False)]]
#             best_point = bone_right_points_exclude_on_crown[0]
#             bone_x.append(best_point[0])
#             bone_y.append(best_point[1])
#         except:
#             pass
# #     print("cej_x:",cej_x,"cej_y:",cej_y)
# #     print("bone_x:",bone_x,"bone_y:",bone_y)
# #     print("-------------------------------")
#     return (cej_x,cej_y),(bone_x,bone_y)

# def remove_pieces(imgarray, mask):
#     iterations_times = 5
#     kernel = np.ones((3,3),np.uint8) 
#     imgarray = imgarray.astype('uint8')
#     mask = mask.astype('uint8')

#     for i in range(iterations_times):
#         imgarray_erosion = cv2.erode(imgarray,kernel,iterations = 2)
#         imgarray_open = cv2.morphologyEx(imgarray_erosion, cv2.MORPH_OPEN, kernel)
    
#         imgarray_dilation = cv2.dilate(imgarray_open,kernel,iterations = 1)
#         imgarray_dilation = imgarray_dilation*mask
#         imgarray_dilation = cv2.dilate(imgarray_dilation,kernel,iterations = 1)
#         imgarray_dilation = imgarray_dilation*mask
#         imgarray = imgarray_dilation
        
#     imgarray_dilation = cv2.dilate(imgarray,kernel,iterations = 1)
#     imgarray_dilation = imgarray_dilation*mask
#     imgarray_dilation = cv2.dilate(imgarray,kernel,iterations = 1)
#     imgarray_dilation = imgarray_dilation*mask
#     return imgarray_dilation*mask

def remove_pieces_300(mask): 
    out_image = np.zeros((mask.shape[0],mask.shape[1]), np.uint8)
    mask_255 = mask*255
    mask_threshold = cv2.threshold(mask_255, 127, 255, cv2.THRESH_BINARY)[1]
    num_labels, labels_im = cv2.connectedComponents(mask_threshold)
    pixel_num_dict = show_pixel_set(labels_im)
    pixel_num_list = list(pixel_num_dict)
    for ele in pixel_num_list[1:]:
        if pixel_num_dict[ele] > 300:
            out_image+=np.where(labels_im==ele,1,0).astype('uint8')
    return out_image

def remove_pieces_50(mask): 
    out_image = np.zeros((mask.shape[0],mask.shape[1]), np.uint8)
    mask_255 = mask*255
    mask_threshold = cv2.threshold(mask_255, 127, 255, cv2.THRESH_BINARY)[1]
    num_labels, labels_im = cv2.connectedComponents(mask_threshold)
    pixel_num_dict = show_pixel_set(labels_im)
    pixel_num_list = list(pixel_num_dict)
    for ele in pixel_num_list[1:]:
        if pixel_num_dict[ele] > 50:
            out_image+=np.where(labels_im==ele,1,0).astype('uint8')
    return out_image

def find_cej_bone_keypoint(img_BGR_array):
    kernel = np.ones((3,3),np.uint8)
    img_gray = cv2.cvtColor(img_BGR_array,cv2.COLOR_BGR2GRAY)
    
    tooth_mask = np.where(img_gray==94,1,0).astype('uint8')
    tooth_mask_rm_p = remove_pieces_300(tooth_mask)
    tooth_contour = cv2.morphologyEx(tooth_mask_rm_p, cv2.MORPH_GRADIENT, kernel)
    
    background_mask = np.where(img_gray==0,1,np.where(img_gray==86,1,np.where(img_gray==151,1,0))).astype('uint8')
    background_mask_rm_p = remove_pieces_50(background_mask)
    background_contour = cv2.morphologyEx(background_mask_rm_p, cv2.MORPH_GRADIENT, kernel)
    
    cej_crown_mask = np.where(img_gray==150,1,np.where(img_gray==179,1,0)).astype('uint8')
    cej_crown_mask_rm_p = remove_pieces_300(cej_crown_mask)

    gums_mask = np.where(img_gray==29,1,0).astype('uint8')
    gums_mask_rm_p = remove_pieces_300(gums_mask)

    cej = tooth_contour*background_contour*cej_crown_mask_rm_p
    bone = tooth_contour*background_contour*gums_mask_rm_p
    
    cej_point = np.nonzero(cej)
    bone_point = np.nonzero(bone)
    return (cej_point[1],cej_point[0]), (bone_point[1],bone_point[0])


def bone_recover(bone_x_list, bone_y_list, fill_up_img, line_of_inertia, seg_img_with_cej_img, value, molar):

    kernel = np.ones((5,5),np.uint8)
    img_gray = cv2.cvtColor(seg_img_with_cej_img,cv2.COLOR_BGR2GRAY)
    tooth = cv2.morphologyEx(np.where(img_gray==94,1,0).astype('uint8'), cv2.MORPH_GRADIENT, kernel)
    gums = cv2.morphologyEx(np.where(img_gray==29,1,0).astype('uint8'), cv2.MORPH_GRADIENT, kernel)
#     background = cv2.morphologyEx(np.where(img_gray==0,1,np.where(img_gray==86,1,np.where(img_gray==151,1,0))).astype('uint8'), cv2.MORPH_GRADIENT, kernel)
    background_mask = np.where(img_gray==0,1,np.where(img_gray==86,1,np.where(img_gray==151,1,0))).astype('uint8')
    background_mask_rm_p = remove_pieces_50(background_mask)
    background = cv2.morphologyEx(background_mask_rm_p, cv2.MORPH_GRADIENT, kernel)
    
#     gg = tooth*background
#     mask = np.expand_dims(abs(1-gg), axis=2)
#     mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)

    
    
    if bone_x_list !=[]:
        bone_x = bone_x_list[0]
        bone_y = bone_y_list[0]
        point_state = point_in_left_side_of_line((bone_x, bone_y), line_of_inertia[0], line_of_inertia[1])
        point_target_state = not point_state

        (gums_y_, gums_x_) = np.nonzero(tooth*gums)
        gum_point_list = [(y, x) for y ,x in zip(gums_y_,gums_x_) if point_in_left_side_of_line((x,y), line_of_inertia[0],line_of_inertia[1]) == point_target_state]
        gum_point_list.sort(key = lambda x: x[0])

        if value == 0:
            try:
                gum_hight = gum_point_list[0][0]
            except:
                gum_hight = seg_img_with_cej_img.shape[0]
        elif value == 1:
            try:
                gum_hight = gum_point_list[-1][0]
            except:
                gum_hight = 0
        
        (bac_y_, bac_x_) = np.nonzero(tooth*background)
        
        if point_target_state:
            root_x, root_y, line_of_inertia = get_molar_root(fill_up_img, value, line_of_inertia, 'L')
        else:
            root_x, root_y, line_of_inertia = get_molar_root(fill_up_img, value, line_of_inertia, 'R')
            
        bac_point_list = [(y, x) for y ,x in zip(bac_y_, bac_x_) if point_in_left_side_of_line((x,y), line_of_inertia[0],line_of_inertia[1]) == point_target_state]
        if value == 0:
            bac_point_list_filter = [point for point in bac_point_list if point[0] < gum_hight]
        elif value == 1:
            bac_point_list_filter = [point for point in bac_point_list if point[0] > gum_hight]
        bac_point_list_filter.sort(key = lambda x: x[0])

        
        if len(bac_point_list_filter)>1:
            if value == 0:
                #(x, y)
                Candidate_point = (bac_point_list_filter[-1][1], bac_point_list_filter[-1][0]+40)
                #(x, y)
                Candidate_cross_point_with_inertia = crosspoint_point2line(Candidate_point ,line_of_inertia[0],line_of_inertia[1])
                #(y, x)
                Candidate_cross_point_with_inertia = Candidate_cross_point_with_inertia[::-1]
                bone_point_distance_list = [distance_2_points(Candidate_cross_point_with_inertia, point) for point in bac_point_list_filter]
                target_point_index = bone_point_distance_list.index(min(bone_point_distance_list))
                return  ([bone_x, bac_point_list_filter[target_point_index][1]], [bone_y, bac_point_list_filter[target_point_index][0]])
            elif value == 1:
                #(x, y)
                Candidate_point = (bac_point_list_filter[0][1], bac_point_list_filter[0][0]-40)
                #(x, y)
                Candidate_cross_point_with_inertia = crosspoint_point2line(Candidate_point ,line_of_inertia[0],line_of_inertia[1])
                #(x, y) -> (y, x)
                Candidate_cross_point_with_inertia = Candidate_cross_point_with_inertia[::-1]
                bone_point_distance_list = [distance_2_points(Candidate_cross_point_with_inertia, point) for point in bac_point_list_filter]
                target_point_index = bone_point_distance_list.index(min(bone_point_distance_list))
                return  ([bone_x, bac_point_list_filter[target_point_index][1]], [bone_y, bac_point_list_filter[target_point_index][0]])
        else:
             return  ([bone_x], [bone_y])
    else:
        (gums_y_, gums_x_) = np.nonzero(tooth*gums)
        root_x, root_y, line_of_inertia_L = get_molar_root(fill_up_img, value, line_of_inertia, 'L')
        gum_point_list_left = [(y, x) for y, x in zip(gums_y_,gums_x_) if point_in_left_side_of_line((x,y), line_of_inertia_L[0],line_of_inertia_L[1]) == True]
        
        root_x, root_y, line_of_inertia_R = get_molar_root(fill_up_img, value, line_of_inertia, 'R')
        gum_point_list_right = [(y, x) for y, x in zip(gums_y_,gums_x_) if point_in_left_side_of_line((x,y), line_of_inertia_R[0],line_of_inertia_R[1]) == False]
        
        gum_point_list_left.sort(key = lambda x: x[0])
        gum_point_list_right.sort(key = lambda x: x[0])
        
        if value == 0:
            try:
                gum_hight_left = gum_point_list_left[0][0]
            except:
                gum_hight_left = seg_img_with_cej_img.shape[0]
            try:
                gum_hight_right = gum_point_list_right[0][0]

            except:
                gum_hight_right = seg_img_with_cej_img.shape[0]
        elif value == 1:
            try:
                gum_hight_left = gum_point_list_left[-1][0]
            except:
                gum_hight_left = 0
            try:
                gum_hight_right = gum_point_list_right[-1][0]

            except:
                gum_hight_right = 0

        (bac_y_, bac_x_) = np.nonzero(tooth*background)
        root_x, root_y, line_of_inertia_L = get_molar_root(fill_up_img, value, line_of_inertia, 'L')
        bac_point_list_left = [(y, x) for y ,x in zip(bac_y_, bac_x_) if point_in_left_side_of_line((x,y), line_of_inertia_L[0],line_of_inertia_L[1]) == True]
        
        root_x, root_y, line_of_inertia_R = get_molar_root(fill_up_img, value, line_of_inertia, 'R')
        bac_point_list_right = [(y, x) for y ,x in zip(bac_y_, bac_x_) if point_in_left_side_of_line((x,y), line_of_inertia_R[0],line_of_inertia_R[1]) == False]
        
        b_l_y = [p[0] for p in bac_point_list_left]
        b_l_x = [p[1] for p in bac_point_list_left]
    
        b_r_y = [p[0] for p in bac_point_list_right]
        b_r_x = [p[1] for p in bac_point_list_right]
        

        if value == 0:
            bac_point_list_left_filter = [point for point in bac_point_list_left if point[0] < gum_hight_left]
            bac_point_list_right_filter = [point for point in bac_point_list_right if point[0] < gum_hight_right]
        elif value == 1:
            bac_point_list_left_filter = [point for point in bac_point_list_left if point[0] > gum_hight_left]
            bac_point_list_right_filter = [point for point in bac_point_list_right if point[0] > gum_hight_right]
            
        b_l_y = [p[0] for p in bac_point_list_left_filter]
        b_l_x = [p[1] for p in bac_point_list_left_filter]
    
        b_r_y = [p[0] for p in bac_point_list_right_filter]
        b_r_x = [p[1] for p in bac_point_list_right_filter]
        
        
        bac_point_list_left_filter.sort(key = lambda x: x[0])
        bac_point_list_right_filter.sort(key = lambda x: x[0])

        bone_x_new = []
        bone_y_new = []
        if len(bac_point_list_left_filter)>0:
            if value == 0:
                #(x, y)
                Candidate_point = (bac_point_list_left_filter[-1][1], bac_point_list_left_filter[-1][0]+40)
                #(x, y)
                Candidate_cross_point_with_inertia = crosspoint_point2line(Candidate_point ,line_of_inertia[0],line_of_inertia[1])
                #(y, x)
                Candidate_cross_point_with_inertia = Candidate_cross_point_with_inertia[::-1]
                bone_point_distance_list = [distance_2_points(Candidate_cross_point_with_inertia, point) for point in bac_point_list_left_filter]
                target_point_index = bone_point_distance_list.index(min(bone_point_distance_list))
                bone_x_new.append(bac_point_list_left_filter[target_point_index][1])
                bone_y_new.append(bac_point_list_left_filter[target_point_index][0])
                
#                 bone_x_new.append(bac_point_list_left_filter[-1][1])
#                 bone_y_new.append(bac_point_list_left_filter[-1][0])
            elif value == 1:
                #(x, y)
                Candidate_point = (bac_point_list_left_filter[0][1], bac_point_list_left_filter[0][0]-40)
                #(x, y)
                Candidate_cross_point_with_inertia = crosspoint_point2line(Candidate_point ,line_of_inertia[0],line_of_inertia[1])
                #(x, y) -> (y, x)
                Candidate_cross_point_with_inertia = Candidate_cross_point_with_inertia[::-1]
                bone_point_distance_list = [distance_2_points(Candidate_cross_point_with_inertia, point) for point in bac_point_list_left_filter]
                target_point_index = bone_point_distance_list.index(min(bone_point_distance_list))

                bone_x_new.append(bac_point_list_left_filter[target_point_index][1])
                bone_y_new.append(bac_point_list_left_filter[target_point_index][0])
            
        if len(bac_point_list_right_filter)>0:
            if value == 0:
                #(x, y)
                Candidate_point = (bac_point_list_right_filter[-1][1], bac_point_list_right_filter[-1][0]+40)
                #(x, y)
                Candidate_cross_point_with_inertia = crosspoint_point2line(Candidate_point ,line_of_inertia[0],line_of_inertia[1])
                #(y, x)
                Candidate_cross_point_with_inertia = Candidate_cross_point_with_inertia[::-1]
                bone_point_distance_list = [distance_2_points(Candidate_cross_point_with_inertia, point) for point in bac_point_list_right_filter]
                target_point_index = bone_point_distance_list.index(min(bone_point_distance_list))
                
                bone_x_new.append(bac_point_list_right_filter[target_point_index][1])
                bone_y_new.append(bac_point_list_right_filter[target_point_index][0])   

            elif value == 1:
                #(x, y)
                Candidate_point = (bac_point_list_right_filter[0][1], bac_point_list_right_filter[0][0]-40)
                #(x, y)
                Candidate_cross_point_with_inertia = crosspoint_point2line(Candidate_point ,line_of_inertia[0],line_of_inertia[1])
                #(x, y) -> (y, x)
                Candidate_cross_point_with_inertia = Candidate_cross_point_with_inertia[::-1]
                bone_point_distance_list = [distance_2_points(Candidate_cross_point_with_inertia, point) for point in bac_point_list_right_filter]
                target_point_index = bone_point_distance_list.index(min(bone_point_distance_list))
                bone_x_new.append(bac_point_list_right_filter[target_point_index][1])
                bone_y_new.append(bac_point_list_right_filter[target_point_index][0])  
                
        return  (bone_x_new, bone_y_new)