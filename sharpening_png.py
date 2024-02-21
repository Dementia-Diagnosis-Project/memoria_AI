# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 23:16:57 2024

@author: user
"""

import os
import numpy as np
import cv2

Data_path='D:\ADNI_process_done_2'
sharp_filter_1=np.array([[-1,-1,-1], [-1,9,-1],[-1,-1,-1]])
sharp_filter_2=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
clinical_stage=[stage for stage in os.listdir(Data_path) if '.zip' not in stage]

for stage in clinical_stage:
    subjects=[subj for subj in os.listdir(os.path.join(Data_path,stage))]
    for subj in subjects:
        os.makedirs(os.path.join('D:\ADNI_sharpening_1',stage, subj))
        os.makedirs(os.path.join('D:\ADNI_sharpening_2',stage, subj))
        os.chdir(os.path.join(Data_path, stage, subj))
        pngs=os.listdir()
        pngs.sort()
        
        for i in range(len(pngs)):
            image='plane'+str(i)+'.png'
            image_org=cv2.imread(image)
            sharp_image_1=cv2.filter2D(image_org,-1,sharp_filter_1)
            sharp_image_2=cv2.filter2D(image_org,-1,sharp_filter_2)
            
            cv2.imwrite(os.path.join('D:\ADNI_sharpening_1', stage, subj, f'sharp1_plane{i}.png'), sharp_image_1)
            cv2.imwrite(os.path.join('D:\ADNI_sharpening_2', stage, subj, f'sharp2_plane{i}.png'), sharp_image_2)
            