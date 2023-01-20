import os
import re
import cv2
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from helper import save_obj, load_obj


def get_rot_tra(rot_adr, tra_adr):
    """
    Helper function to the read the rotation and translation file
        Args:
                rot_adr (str): path to the file containing rotation of an object
        tra_adr (str): path to the file containing translation of an object
        Returns:
                rigid transformation (np array): rotation and translation matrix combined
    """

    rot_matrix = np.loadtxt(rot_adr, skiprows=1)
    trans_matrix = np.loadtxt(tra_adr, skiprows=1)
    trans_matrix = np.reshape(trans_matrix, (3, 1))
    rigid_transformation = np.append(rot_matrix, trans_matrix, axis=1)

    return rigid_transformation


def fill_holes(idmask, umask, vmask):
    """
    Helper function to fill the holes in id , u and vmasks
        Args:
                idmask (np.array): id mask whose holes you want to fill
        umask (np.array): u mask whose holes you want to fill
        vmask (np.array): v mask whose holes you want to fill
        Returns:
                filled_id_mask (np array): id mask with holes filled
        filled_u_mask (np array): u mask with holes filled
        filled_id_mask (np array): v mask with holes filled
    """
    idmask = np.array(idmask, dtype='float32')
    umask = np.array(umask, dtype='float32')
    vmask = np.array(vmask, dtype='float32')
    thr, im_th = cv2.threshold(idmask, 0, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    res = cv2.morphologyEx(im_th, cv2.MORPH_OPEN, kernel)
    im_th = cv2.bitwise_not(im_th)
    des = cv2.bitwise_not(res)
    mask = np.array(des-im_th, dtype='uint8')
    filled_id_mask = cv2.inpaint(idmask, mask, 5, cv2.INPAINT_TELEA)
    filled_u_mask = cv2.inpaint(umask, mask, 5, cv2.INPAINT_TELEA)
    filled_v_mask = cv2.inpaint(vmask, mask, 5, cv2.INPAINT_TELEA)

    return filled_id_mask, filled_u_mask, filled_v_mask


def create_GT_masks(root_dir, background_dir, intrinsic_matrix,classes):
    """
    Helper function to create the Ground Truth ID,U and V masks
        Args:
        root_dir (str): path to the root directory of the dataset
        background_dir(str): path t
        intrinsic_matrix (array): matrix containing camera intrinsics
        classes (dict) : dictionary containing classes and their ids
        Saves the masks to their respective directories
    """
    list_all_images = load_obj(root_dir + "all_images_adr") # 모든 image들의 주소
    training_images_idx = load_obj(root_dir + "train_images_indices") # dataset에서 color image number
    
    for i in range(len(training_images_idx)):
        img_adr = list_all_images[training_images_idx[i]]
        label = os.path.split(os.path.split(os.path.dirname(img_adr))[0])[1]
        regex = re.compile(r'\d+')
        idx = regex.findall(os.path.split(img_adr)[1])[0]

        if i % 1000 == 0:
            print(str(i) + "/" + str(len(training_images_idx)) + " finished!")

        # image 불러옴
        image = cv2.imread(img_adr)
        # image.shape 
        
        # image shape 출력해보기
        num=0
        if num==0:
            print('height: ',image.shape[0])
            print('width:', image.shape[1])
        num+=1
        # image와 동일한 사이즈로 만들어두기
        ID_mask = np.zeros((image.shape[0], image.shape[1])) # height, width
        U_mask = np.zeros((image.shape[0], image.shape[1]))
        V_mask = np.zeros((image.shape[0], image.shape[1]))

        ID_mask_file = root_dir + label + \
            "/ground_truth/IDmasks/color" + str(idx) + ".png"
        U_mask_file = root_dir + label + \
            "/ground_truth/Umasks/color" + str(idx) + ".png"
        V_mask_file = root_dir + label + \
            "/ground_truth/Vmasks/color" + str(idx) + ".png"

        tra_adr = root_dir + label + "/data/tra" + str(idx) + ".tra" # 각 이미지의 translation 정보
        rot_adr = root_dir + label + "/data/rot" + str(idx) + ".rot" # 각 이미지의 rotation 정보
        # translation과 rotation 정보 -> transformation matrix 생성 (rigid transformation)
        rigid_transformation = get_rot_tra(rot_adr, tra_adr)

        # Read point Point Cloud Data
        # 3차원 model의 표면에 있는 점들의 집합
        ptcld_file = root_dir + label + "/object.xyz"
        pt_cld_data = np.loadtxt(ptcld_file, skiprows=1, usecols=(0, 1, 2))
        ones = np.ones((pt_cld_data.shape[0], 1))
        
        # homogenous_coordinate: 동차좌표계  (x,y)를 (x,y,1)로 표현하는 것  
        homogenous_coordinate = np.append(pt_cld_data[:, :3], ones, axis=1)

        # Perspective Projection to obtain 2D coordinates for masks
        homogenous_2D = intrinsic_matrix @ (                # @는 내적?
            rigid_transformation @ homogenous_coordinate.T)
        coord_2D = homogenous_2D[:2, :] / homogenous_2D[2, :]
        coord_2D = ((np.floor(coord_2D)).T).astype(int)
        x_2d = np.clip(coord_2D[:, 0], 0, 639)  # np.clip(value_array, min, max) 전달받은 숫자가 최소, 최대값 사이의 값인지 확인
        y_2d = np.clip(coord_2D[:, 1], 0, 479)  
        ID_mask[y_2d, x_2d] = classes[label]

        if i % 100 != 0:  # change background for every 99/100 images
            background_img_adr = background_dir + \
                random.choice(os.listdir(background_dir))
            background_img = cv2.imread(background_img_adr)
            background_img = cv2.resize(
                background_img, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
            background_img[y_2d, x_2d, :] = image[y_2d, x_2d, :]
            background_adr = root_dir + label + \
                "/changed_background/color" + str(idx) + ".png"
            mpimg.imsave(background_adr, background_img)

        # Generate Ground Truth UV Maps
        centre = np.mean(pt_cld_data, axis=0)
        length = np.sqrt((centre[0]-pt_cld_data[:, 0])**2 + (centre[1] -
                                                             pt_cld_data[:, 1])**2 + (centre[2]-pt_cld_data[:, 2])**2)
        unit_vector = [(pt_cld_data[:, 0]-centre[0])/length, (pt_cld_data[:,
                                                                          1]-centre[1])/length, (pt_cld_data[:, 2]-centre[2])/length]
        U = 0.5 + (np.arctan2(unit_vector[2], unit_vector[0])/(2*np.pi))
        V = 0.5 - (np.arcsin(unit_vector[1])/np.pi)
        U_mask[y_2d, x_2d] = U
        V_mask[y_2d, x_2d] = V

        # Saving ID, U and V masks after using the fill holes function
        ID_mask, U_mask, V_mask = fill_holes(ID_mask, U_mask, V_mask)
        cv2.imwrite(ID_mask_file, ID_mask)
        mpimg.imsave(U_mask_file, U_mask, cmap='gray')
        mpimg.imsave(V_mask_file, V_mask, cmap='gray')


def create_UV_XYZ_dictionary(root_dir):

    classes = ['ape', 'benchviseblue', 'can', 'cat', 'driller', 'duck', 'glue', 'holepuncher',
               'iron', 'lamp', 'phone', 'cam', 'eggbox']
    # create a dictionary for UV to XYZ correspondence
    for label in classes:
        ptcld_file = root_dir + label + "/object.xyz"
        pt_cld_data = np.loadtxt(ptcld_file, skiprows=1, usecols=(0, 1, 2))
        # calculate u and v coordinates from the xyz point cloud file
        centre = np.mean(pt_cld_data, axis=0)
        length = np.sqrt((centre[0]-pt_cld_data[:, 0])**2 + (centre[1] -
                                                             pt_cld_data[:, 1])**2 + (centre[2]-pt_cld_data[:, 2])**2)
        unit_vector = [(pt_cld_data[:, 0]-centre[0])/length, (pt_cld_data[:,
                                                                          1]-centre[1])/length, (pt_cld_data[:, 2]-centre[2])/length]
        u_coord = 0.5 + (np.arctan2(unit_vector[2], unit_vector[0])/(2*np.pi))
        v_coord = 0.5 - (np.arcsin(unit_vector[1])/np.pi)
        u_coord = (u_coord * 255).astype(int)
        v_coord = (v_coord * 255).astype(int)
        # save the mapping as a pickle file
        dct = {}
        for u, v, xyz in zip(u_coord, v_coord, pt_cld_data):
            key = (u, v)
            if key not in dct:
                dct[key] = xyz
        save_obj(dct, root_dir + label + "/UV-XYZ_mapping")


def dataset_dir_structure(root_dir):

    classes = ['ape', 'benchviseblue', 'can', 'cat', 'driller', 'duck', 'glue', 'holepuncher',
               'iron', 'lamp', 'phone', 'cam', 'eggbox']

    for label in classes:  # create directories to store data
        os.mkdir(root_dir + label + "/predicted_pose")
        os.mkdir(root_dir + label + "/ground_truth")
        os.mkdir(root_dir + label + "/ground_truth/IDmasks")
        os.mkdir(root_dir + label + "/ground_truth/Umasks")
        os.mkdir(root_dir + label + "/ground_truth/Vmasks")
        os.mkdir(root_dir + label + "/changed_background")
        os.mkdir(root_dir + label + "/pose_refinement")
        os.mkdir(root_dir + label + "/pose_refinement/real")
        os.mkdir(root_dir + label + "/pose_refinement/rendered")
