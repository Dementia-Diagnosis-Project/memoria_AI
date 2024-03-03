# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 00:10:08 2024

@author: Park jieun
"""
#!pip install SimpleITK==1.2.0
#!pip install SimpleITK-SimpleElastix
#!pip install dicom2nifti
#install ANTsPY
#pip install antspynet

#import SimpleITK as sitk
import os
import sys
import zipfile
import shutil
import dicom2nifti
import glob
import gzip
import ants
import numpy as np
import nibabel as nib
from PIL import Image
from antspynet.utilities import brain_extraction

def arrange_folders_and_convert(Data_path):
    os.chdir(Data_path)
    zip_files=os.listdir()
    zip_files=[files for files in zip_files if '.zip' in files]
    
    # zip 파일 압축해제
    for zip_file in zip_files:
        zipfile.ZipFile(os.path.join(Data_path,zip_file)).extractall()
        # 압축해제하면 폴더 이름 'ADNI'로 나와서 원래 zip파일 이름으로 바꿔줌 (ex: MCI.zip -> MCI)
        os.rename('ADNI',zip_file[:-4])
        os.chdir(os.path.join(Data_path, zip_file[:-4]))
        
        # subject 번호 list 
        subjects=os.listdir()
        for subject in subjects:
            os.chdir(os.path.join(Data_path, zip_file[:-4], subject))
            
            # MPRAGE 폴더 list
            MPRAGE_folders=os.listdir()
            for MPRAGE_folder in MPRAGE_folders:
                os.chdir(os.path.join(Data_path, zip_file[:-4],subject,MPRAGE_folder))
                
                # 날짜 폴더 list -> 여기서 subject 번호랑 결합하여 제일 앞 경로에 새 폴더 만듦. (ex: subject번호_240126)
                subfolders=os.listdir()
                                
                for subfolder in subfolders:                
                    folder_name=subject+"_"+subfolder[2:4]+subfolder[5:7]+subfolder[8:10]
                    os.chdir(os.path.join(Data_path, zip_file[:-4],subject,MPRAGE_folder,subfolder))
                    
                    # dicom 도달하기 진짜 마지막 폴더 (사람마다 폴더 이름이 달라서 list로 함)
                    last_folder=[folder for folder in os.listdir()]
                    os.chdir(last_folder[0])
                    os.mkdir(os.path.join(Data_path, zip_file[:-4], folder_name))
                    Data_dir=os.getcwd()
                    
                    # dicom 도달, 새로 만든 폴더 (subject번호_날짜)에 변환한 nifti파일 바로 옮기기
                    dicom2nifti.convert_directory(Data_dir, os.path.join(Data_path, zip_file[:-4], folder_name))
                    
                    # 변환한 nifti 파일은 gzip 파일로 되어있음 (일종의 압축 파일, nii.gz). nii.gz를 nii로 압축해제 
                    os.chdir(os.path.join(Data_path, zip_file[:-4], folder_name))
                    
                    niftigz_file=os.listdir()
                    niftigz_file=[nifti for nifti in niftigz_file if '.nii.gz' in nifti]
                    nifti_file=niftigz_file[0][:-3]
                    with gzip.open(niftigz_file[0], 'rb') as f_in:
                        nii_content = f_in.read()
                    with open(nifti_file, 'wb') as f_out:
                        f_out.write(nii_content)
                    
                    # 압축해제한 nifti 파일 이름을 (subject번호_날짜)로 변경 및 dicom 파일 포함 이전 경로 삭제
                    os.rename(nifti_file, folder_name + '.nii')
                    os.remove(os.path.join(Data_path, zip_file[:-4], folder_name, niftigz_file[0]))
                    
                    [os.remove(f) for f in glob.glob(os.path.join(Data_path, zip_file[:-4], folder_name, '*.dcm'))]
                    
            os.chdir(Data_path)                
            shutil.rmtree(os.path.join(Data_path, zip_file[:-4], subject))
                    
                    


"""
# MNI template는 MRI 상위폴더(subject 모아놓은 폴더)랑 같은 위치에 있는 걸로 보면 됨.
def registeration_ITK(Data_path, anat_path, nifti_file, MNI_152='MNI152_T1_1mm.nii'):
    T1_nifti=sitk.ReadImage(os.path.join(anat_path, nifti_file))
    MNI_nifti=sitk.ReadImage(os.path.join(Data_path, MNI_152))
    parameter=sitk.GetDefaultParameterMap('rigid')


    elastixImageFilter = sitk.ElastixImageFilter()

    elastixImageFilter.SetFixedImage(MNI_nifti)
    elastixImageFilter.SetMovingImage(T1_nifti)
    elastixImageFilter.SetParameterMap(parameter)

    elastixImageFilter.Execute()
    sitk.WriteImage(elastixImageFilter.GetResultImage(), os.path.join(anat_path, 'result.nii'))
    sitk.WriteParameterFile(elastixImageFilter.GetTransformParameterMap()[0], os.path.join(anat_path, 'T1orig_parameter.txt'))
"""

def registration_ANT(subj_path, nifti_file, MNI_152, registered_T1_name, transform, hippocampus):
    T1_nifti=ants.image_read(os.path.join(subj_path,nifti_file))
    MNI_nifti=ants.image_read(os.path.join(Data_path, MNI_152))
    
    transformation=ants.registration(fixed=MNI_nifti, moving=T1_nifti, type_of_transform=transform)
    registered_T1=transformation['warpedmovout']
    registered_T1.to_file(os.path.join(subj_path, registered_T1_name))
    
    if hippocampus != None:
        
        BI_HP=ants.image_read(os.path.join(subj_path, hippocampus))
        FS_image=ants.image_read(os.path.join(subj_path, 'brain.nii'))
        
        FS_transformation=ants.registration(fixed=T1_nifti, moving=FS_image, type_of_transform=transform)
        HP_image=ants.apply_transforms(fixed=T1_nifti, moving=BI_HP, transformlist=FS_transformation['fwdtransforms'], interpolator='nearestNeighbor')
        HP_name='reg_HP_' + subj + '.nii'
        HP_image.to_file(os.path.join(subj_path, HP_name))
        
        HP=nib.load(os.path.join(subj_path,hippocampus))
        HP_array=HP.get_fdata()
        HP_mask=(HP_array >= 0).astype(int)
        HP_mask_nii=nib.Nifti1Image(HP_mask, HP.affine)
        
        os.remove(os.path.join(subj_path, HP_name))
        nib.save(HP_mask_nii, os.path.join(subj_path, HP_name))
        
        
    

def skull_stripping_with_biascorrection(subj_path, reg_nifti):
    regT1_nifti=regT1_nifti=ants.image_read(os.path.join(subj_path, reg_nifti))
    
    prob_brain_mask=brain_extraction(regT1_nifti, modality='t1')
    brain_mask=ants.get_mask(prob_brain_mask, low_thresh=0.5)
    masked=ants.mask_image(regT1_nifti, brain_mask)
    biascorrected_mask=ants.n4_bias_field_correction(masked)
    
    biascorrected_mask.to_file(os.path.join(subj_path, brain_mask_name))
    
    
    
    
    
Data_path='D:\ADNI_subjects' #sys.argv[1] #D:\ADNI_subjects
arrange_folders_and_convert(Data_path)

clinical_groups=os.listdir()
clinical_groups=[group for group in clinical_groups if ('.zip' not in group) and ('excel_files' not in group) and ('MNI' not in group)]

for clinical_group in clinical_groups:
    os.chdir(os.path.join(Data_path, clinical_group))
    
    subjects=os.listdir()
    for subj in subjects:
        subjects_path=os.path.join(Data_path, clinical_group, subj)
        subject_nifti=os.listdir(subjects_path)
        
        brain_mask_name='brain_' + subj + '.nii'
        
        registration_ANT(subjects_path, subject_nifti[0], 'MNI152_T1_1mm.nii', 'reg_' + subj + '.nii', 'Rigid', None)
        skull_stripping_with_biascorrection(subjects_path, 'reg_' + subj + '.nii')
        #registration_ANT(subjects_path, brain_mask_name, 'MNI152_T1_1mm_brain.nii', 'final_' + subj + '.nii', 'BI_HP.nii')
        registration_ANT(subjects_path, brain_mask_name, 'MNI152_T1_1mm_brain.nii', 'final_' + subj + '.nii', 'Rigid', None)
        
        reg_T1=nib.load(os.path.join(subjects_path, 'final_' + subj + '.nii'))
        reg_T1_array=reg_T1.get_fdata()
        
        start_slice=90
        end_slice=120
        selected_slice=reg_T1_array[:, start_slice:end_slice, :]
        
        sliced_T1=nib.Nifti1Image(selected_slice, reg_T1.affine)
        nib.save(sliced_T1, os.path.join(subjects_path, 'sliced_final_' + subj + '.nii'))
        scan=sliced_T1.get_fdata()
        normalized_scan=((scan - np.min(scan)) / (np.max(scan) - np.min(scan)) * 255).astype(np.uint8)
        normalized_nifti=nib.Nifti1Image(normalized_scan, sliced_T1.affine)
        nib.save(normalized_nifti, os.path.join(subjects_path, 'norm_sliced_final' + subj + '.nii'))
        for i in range(normalized_scan.shape[1]):
                img = Image.fromarray(normalized_scan[:, i, :].T)  
                img = img.rotate(180)
                img.save(os.path.join(subjects_path, f'plane{i}.png'))
        
        
        

                    