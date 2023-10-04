'''
 @FileName    : dataset.py
 @EditTime    : 2022-09-27 16:03:55
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

import os
import torch
import numpy as np
import cv2
from utils.geometry import estimate_translation_np
from utils.imutils import crop, flip_img, flip_pose, flip_kp, surface_projection, transform, rot_aa
from datasets.base import base
import constants

from yolox.yolox import Predictor
import cv2

class DemoData(base):
    def __init__(self, train=True, dtype=torch.float32, data_folder='', name='', smpl=None):
        super(DemoData, self).__init__(train=train, dtype=dtype, data_folder=data_folder, name=name, smpl=smpl)

        self.max_people = 8
        self.eval = True

        self.imnames = [os.path.join(self.dataset_dir, img) for img in os.listdir(self.dataset_dir)]
        self.annot = []
        self.intris = []
        self.len = len(self.imnames)

    def human_detection(self,):
        model_dir = R'data/bytetrack_x_mot17.pth.tar'
        thres = 0.23
        predictor = Predictor(model_dir, thres)

        for img in self.imnames:
            # GT intri for JTA
            self.intris.append(np.array([[1158.,0.,960.],[0.,1158.,540.],[0.,0.,1.]]))
            results, result_img = predictor.predict(img, viz=False)
            self.annot.append(results)

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        rgb_img, ul, br, new_shape, new_x, new_y, old_x, old_y = crop(rgb_img, center, scale, 
                      [constants.IMG_RES, constants.IMG_RES], rot=rot)
        # flip the image 
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img, ul, br, new_shape, new_x, new_y, old_x, old_y

    def estimate_trans_cliff(self, joints, keypoints, center, focal_length, img_h, img_w):
        keypoints = keypoints.clone().detach().numpy()
        joints = joints.detach().numpy()
        keypoints[:,:-1] = keypoints[:,:-1] * constants.IMG_RES + np.array(center)
        
        gt_cam_t = estimate_translation_np(joints, keypoints[:,:2], keypoints[:,2], focal_length=focal_length, center=[img_w/2, img_h/2])
        return gt_cam_t
    
    # Data preprocess
    def create_data(self, index=0):
        load_data = {}
        annot = self.annot[index]
        imgname = self.imnames[index]
        intris = self.intris[index]
        
        # Load image
        try:
            img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)
        except TypeError:
            print(imgname)
        img_h, img_w, _ = img.shape
        load_data["imgname"] = imgname

        batch_size = len(annot['bbox']) // self.max_people + 1

        norm_imgs = torch.zeros((batch_size, self.max_people, 3, 224, 224)).float()
        centers = torch.zeros((batch_size, self.max_people, 2)).float()
        scales = torch.zeros((batch_size, self.max_people)).float()
        crop_uls = torch.zeros((batch_size, self.max_people, 2)).float()
        crop_brs = torch.zeros((batch_size, self.max_people, 2)).float()
        img_hs = torch.zeros((batch_size, self.max_people)).float()
        img_ws = torch.zeros((batch_size, self.max_people)).float()
        focal_lengthes = torch.ones((batch_size, self.max_people)).float()
        valid = torch.zeros((batch_size, self.max_people)).float()

        for batch_idx in range(batch_size):
            boxes = annot['bbox'][batch_idx * self.max_people:(batch_idx + 1) * self.max_people]
            num_people = len(boxes)
            
            for i in range(len(boxes)):
                if i >= num_people:
                    break
                valid[batch_idx][i] = 1.

                if intris is not None:
                    focal_length = intris[0][0]
                else:
                    focal_length = (img_h ** 2 + img_w ** 2) ** 0.5

                bbox = boxes[i]

                center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                scale = 1.0*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200.

                patch, crop_ul, crop_br, new_shape, new_x, new_y, old_x, old_y = self.rgb_processing(img.copy(), center, 1*scale, 0, 0, np.ones(3))
                patch = torch.from_numpy(patch).float()
                norm_img = self.normalize_img(patch)

                norm_imgs[batch_idx][i] = norm_img
                centers[batch_idx][i] = torch.from_numpy(np.array(center)).float()
                scales[batch_idx][i] = scale
                crop_uls[batch_idx][i] = torch.from_numpy(crop_ul).float()
                crop_brs[batch_idx][i] = torch.from_numpy(crop_br).float()
                img_hs[batch_idx][i] = img_h
                img_ws[batch_idx][i] = img_w
                focal_lengthes[batch_idx][i] = focal_length


            load_data['valid'] = valid
            load_data["img"] = norm_imgs
            load_data["center"] = centers
            load_data["scale"] = scales
            load_data["crop_ul"] = crop_uls
            load_data["crop_br"] = crop_brs
            load_data["img_h"] = img_hs
            load_data["img_w"] = img_ws
            load_data["focal_length"] = focal_lengthes


        return load_data

    def __getitem__(self, index):
        data = self.create_data(index)
        return data

    def __len__(self):
        return self.len













