import tensorflow as tf
import albumentations as A
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_patient_info(patient_list, file_list):
    print('There {} patient in this folder.\n'.format(len(patient_list)))
    for index, patient in enumerate(patient_list):
        print('{}.{} \t'.format(index+1,patient.split('/')[-1]))
        count = 0
        for data in file_list:
            if patient in data:
                count+=1
        print('Having {} data in folder.\n'.format(count))



# from collections import Counter

# def ensemble_three_class(img, model, flip_ = False):
#     pr_img = model.predict(img)
#     sqpr_img = np.squeeze(pr_img)
#     if flip_:
#         img = np.squeeze(img)
        
#         h_flip = cv2.flip(img, 1)
#         h_flip = np.expand_dims(h_flip, axis=(0))
#         pre_h_flip = model.predict(h_flip)
#         pre_h_flip = np.squeeze(pre_h_flip)
#         h_flip = cv2.flip(pre_h_flip, 1)
 
#         v_flip = cv2.flip(img, 0)
#         v_flip = np.expand_dims(v_flip, axis=(0))
#         pre_v_flip = model.predict(v_flip)
#         pre_v_flip = np.squeeze(pre_v_flip)
#         v_flip = cv2.flip(pre_v_flip, 0)
        
#         hv_flip = cv2.flip(img, -1)
#         hv_flip = np.expand_dims(hv_flip, axis=(0))
#         pre_hv_flip = model.predict(hv_flip)
#         pre_hv_flip = np.squeeze(pre_hv_flip)
#         hv_flip = cv2.flip(pre_hv_flip, -1)
        
#         pixel_dict = {'0':np.array([1,0,0], dtype='float32'),
#                       '1':np.array([0,1,0], dtype='float32'),
#                       '2':np.array([0,0,1], dtype='float32')}
#         ensemble_img = np.zeros((sqpr_img.shape[0], sqpr_img.shape[1],3), dtype = 'float32')
#         for i in range(sqpr_img.shape[1]):
#             for j in range(sqpr_img.shape[0]):
#                 pixel_valus = str(np.argmax(sqpr_img[i, j, :]))+str(np.argmax(sqpr_img[i, j, :]))+str(np.argmax(h_flip[i, j, :]))+str(np.argmax(v_flip[i, j, :]))+str(np.argmax(hv_flip[i, j, :]))
#                 ensemble_img[i][j] = pixel_dict[Counter(pixel_valus).most_common(1)[0][0]]
#     return np.expand_dims(ensemble_img, axis=0)

# def ensemble_bin(img, model_list, flip_ = False):
#     ensemble_final_result = np.zeros((img.shape[0], img.shape[1],1), dtype = 'float32')
#     for index, model in enumerate(model_list):
#         pr_img = model.predict(img)
#         sqpr_img = np.squeeze(pr_img)
#         if flip_:
#             imgp = np.squeeze(img)
#             h_flip = cv2.flip(imgp, 1)
#             h_flip = np.expand_dims(h_flip, axis=(0,3))
#             pre_h_flip = model.predict(h_flip)
#             pre_h_flip = np.squeeze(pre_h_flip)
#             h_flip = cv2.flip(pre_h_flip, 1)

#             v_flip = cv2.flip(imgp, 0)
#             v_flip = np.expand_dims(v_flip, axis=(0,3))
#             pre_v_flip = model.predict(v_flip)
#             pre_v_flip = np.squeeze(pre_v_flip)
#             v_flip = cv2.flip(pre_v_flip, 0)

#             hv_flip = cv2.flip(imgp, -1)
#             hv_flip = np.expand_dims(hv_flip, axis=(0,3))
#             pre_hv_flip = model.predict(hv_flip)
#             pre_hv_flip = np.squeeze(pre_hv_flip)
#             hv_flip = cv2.flip(pre_hv_flip, -1)
#             ensemble_img = (sqpr_img+h_flip+v_flip+hv_flip)/4
#         ensemble_final_result = ensemble_final_result+ensemble_img
#     ensemble_final_result = ensemble_final_result/(index+1)
#     return np.expand_dims(np.where(ensemble_final_result > 0.5, 255, 0), axis=0)