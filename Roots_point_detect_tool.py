import os
import cv2
import csv
import math
import glob
import random
import shutil
import itertools
import scipy.misc
import numpy as np

from tqdm import tqdm
from matplotlib import pyplot as plt

from scipy.spatial.distance import cdist
from pattern_detection_tool import*


def remove_check_point(path):
    for file_name in os.listdir(path):
        #print(file_name)
        if 'checkpoints' in file_name:
            shutil.rmtree(os.path.join(path, file_name))
            
def folder_builder(parent_path,folder_name,Rebuild):
    if os.path.exists(os.path.join(parent_path, folder_name)) and not Rebuild:
        print('"{}" folder is exist,if want to rebuild please modify "Rebuild" parameter.'.format(folder_name))
    else:
        if os.path.exists(os.path.join(parent_path, folder_name)):
            shutil.rmtree(os.path.join(parent_path, folder_name))
        try:
            os.makedirs(os.path.join(parent_path, folder_name))
            print('"{}" folder build successful'.format(folder_name))
        except:
            print('folder build error')

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = csv.writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

def show_pixel_set(img_nparray):
    a = img_nparray
    unique, counts = np.unique(a, return_counts=True)
    return dict(zip(unique, counts))

def get_name_from_path(path,Extension=True):
    name = os.path.basename(path)
    if Extension:
        return name
    else:
        return name.split('.')[0]

def image_components(img_path,nparray=False):
    img = img_path if nparray else cv2.imread(img_path,0)
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    
    num_labels, labels_im = cv2.connectedComponents(img)
    label_hue = np.uint8(179*labels_im/np.max(labels_im))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
    return labeled_img

def image_split_components(img_path,nparray=False):
    img = img_path if nparray else cv2.imread(img_path,0)
    img = cv2.threshold(img, 0, 127, cv2.THRESH_BINARY)[1]

    num_labels, labels_im = cv2.connectedComponents(img)
    label_hue = np.uint8(179*labels_im/np.max(labels_im))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
    
    gray_components = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)
    pixel_dict = show_pixel_set(gray_components)
    key_list = list(pixel_dict.keys()) 
    
    components_list = []
    for index, pixel in enumerate(key_list):
        if len(key_list)==1:
            img_gray = gray_components.copy()
            component = np.where(img_gray == pixel, 255, 0.)
            components_list.append(component)
        elif index != 0 :
            img_gray = gray_components.copy()
            component = np.where(img_gray == pixel, 255, 0.)
            components_list.append(component)
    return components_list

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

def img_dilation(imgarray,mask,iterations_times):
    kernel = np.ones((3,3),np.uint8) 
    imgarray = imgarray.astype('uint8')
    mask = mask.astype('uint8')
    if iterations_times == 0:
        return imgarray*mask
#     x_v1, y_v1, x_mean, y_mean,len_non_zero = get_inertia_parameter(imgarray)
    for i in range(iterations_times):
        dilation = cv2.dilate(imgarray,kernel,iterations = 1)
        imgarray = dilation*mask
    return imgarray*mask

   
    
def slope(x1, y1, x2, y2):
    try:
        m = (y2-y1)/(x2-x1)
        return abs(m)
    except:
        return None


# def draw_approx_hull_polygon(img):
#     gray = img
#     ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     black_img = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
#     hulls = [cv2.convexHull(cnt) for cnt in contours]
#     cv2.polylines(black_img, hulls, True, (255,0,0), 5)
#     return black_img

def shift_img(img,x,y):
    img = img
    img_size = img.shape
    M = np.float32([[1,0,x],[0,1,y]])
    shifted_img = cv2.warpAffine(img,M,(img_size[1],img_size[0]))
    return shifted_img

def extend_3direction_img(extend_img,mask_img,value,iteration=1):
    img = extend_img
    for i in range(iteration):
        img = img*mask_img
        shifted_img_n10 = shift_img(img,-2,0)
        shifted_img_p10 = shift_img(img,2,0)
        if value==1:
            shifted_img_v10 = shift_img(img,0,-5)
        elif value==0:
            shifted_img_v10 = shift_img(img,0,5)
        img = np.where(shifted_img_n10+shifted_img_p10+shifted_img_v10 > 2,255,0).astype('uint8')
    return img

def combine_seg_cej(seg_img,root_img):
    root_img = np.expand_dims(root_img,axis=2)
    target = np.stack((root_img,root_img,root_img),axis=2)
    target = np.squeeze(target)
    purple = np.array([240, 32, 160])*(root_img)
    seg_img = seg_img*(1-target)+purple
    return seg_img

def get_width(img_array):
    nums = 60
    y, x = np.nonzero(img_array)
    sort_width_list = sorted([(y_,list(y).count(y_)) for y_ in set(y)], reverse=True ,key=lambda x:x[1])[:nums]
    width = int(sum(value[1] for value in sort_width_list)/nums)
    return width

def get_hight(img_array):
    y, x = np.nonzero(img_array)
    hight = max(y) - min(y)
    return hight

def get_root(img_array,value):
    num = 10
    y, x = np.nonzero(img_array)
    if value == 1:
        reverse_ = False
    elif value == 0:
        reverse_ = True
    sort_width_list = sorted([(y_,list(y).count(y_)) for y_ in set(y)], reverse=reverse_ ,key=lambda x:x[1])[:nums]
    width = int(sum(value[1] for value in sort_width_list)/nums)
    return hight


def biggest_components(mask, root_mask): 
    out_image = np.zeros((mask.shape[0],mask.shape[1]), np.uint8).astype('uint8')
    mask_255 = mask*255
    mask_threshold = cv2.threshold(mask_255, 127, 255, cv2.THRESH_BINARY)[1]
    num_labels, labels_im = cv2.connectedComponents(mask_threshold)
    pixel_num_dict = show_pixel_set(labels_im)
    key_list = list(pixel_num_dict.keys())[1:]
    for ele in key_list:
        target_mask = np.where(labels_im==ele, 1, 0).astype('uint8')
        if (np.any((target_mask*root_mask)==1)):
            out_image+=target_mask
    try:
        return out_image
    except:
        return mask
    

def remove_pieces_1000(mask): 
    out_image = np.zeros((mask.shape[0],mask.shape[1]), np.uint8)
    mask_255 = mask*255
    mask_threshold = cv2.threshold(mask_255, 127, 255, cv2.THRESH_BINARY)[1]
    num_labels, labels_im = cv2.connectedComponents(mask_threshold)
    pixel_num_dict = show_pixel_set(labels_im)
    pixel_num_list = list(pixel_num_dict)
    for ele in pixel_num_list[1:]:
        if pixel_num_dict[ele] > 1000:
            out_image+=np.where(labels_im==ele,1,0).astype('uint8')
    return out_image
    
def get_boundary_line_by_inertia(root_mask, tooth_mask):
    root_x, root_y,line_of_inertia = root_point_detect(root_mask,1)
    blank_image = np.zeros((root_mask.shape[0],root_mask.shape[1]), np.uint8)
    cv2.line(blank_image, line_of_inertia[0], line_of_inertia[1], (255), 5)
    y_, x_ = np.nonzero(blank_image)
    inertia_point_list = [(x,y) for x, y in zip(x_,y_)]
    inertia_point_list = list(sorted(inertia_point_list, key = lambda x:x[1]))
    min_x = min(inertia_point_list[0][0],inertia_point_list[-1][0])
    max_x = max(inertia_point_list[0][0],inertia_point_list[-1][0])
    final_mask = np.zeros((root_mask.shape[0],root_mask.shape[1]), np.uint8)
    for i in range(-max_x, root_mask.shape[1]-min_x, 1):
        line_n = np.concatenate((blank_image[:,-i:], blank_image[:,:-i]), axis=1)
        if (np.any(line_n*root_mask==1)):
            final_mask+=line_n
    and_mask = np.where(final_mask*tooth_mask > 0,1,0).astype('uint8')
    target_mask = biggest_components(and_mask, np.where(root_mask > 0,1,0).astype('uint8'))
    target_mask = remove_pieces_1000(target_mask)
    return target_mask