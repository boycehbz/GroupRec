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
from utils.imutils import get_crop, keyp_crop2origin, surface_projection, img_crop2origin
from datasets.base import base
from datasets.relation_feature_data import Relation_Feature_Data

class Relation_Img_Data(Relation_Feature_Data):
    def __init__(self, train=True, dtype=torch.float32, data_folder='', name='', smpl=None):
        super(Relation_Img_Data, self).__init__(train=train, dtype=dtype, data_folder=data_folder, name=name, smpl=smpl)

        self.len = len(self.features)
    
    # Data preprocess
    def create_data(self, index=0):
        
        load_data = {}
        
        # Get augmentation parameters
        flip, pn, rot, sc, gt_input = self.augm_params()
        flip, rot, sc, gt_input = 0, 0, 1, 0
        
        imgname = os.path.join(self.dataset_dir, self.imnames[index])
        img_h, img_w = self.img_size[index]

        num_people = len(self.features[index])

        imgnames = ['empty'] * self.max_people
        valid = np.zeros((self.max_people), dtype=np.float32)
        has_3d = np.zeros(self.max_people, dtype=np.float32)
        has_smpls = np.zeros(self.max_people, dtype=np.float32)
        poses = torch.zeros((self.max_people, 72)).float()
        shapes = torch.zeros((self.max_people, 10)).float()
        vertss = torch.zeros((self.max_people, 6890, 3)).float()
        if self.dataset_name in ['Panoptic']:
            gt_joints = torch.zeros((self.max_people, 17, 4)).float()
        else:
            gt_joints = torch.zeros((self.max_people, 26, 4)).float()
        gt_trans = torch.zeros((self.max_people, 3)).float()
        pose2d_gt = torch.zeros((self.max_people, 26, 3)).float()
        imgs = torch.zeros((self.max_people, 3, 224, 224)).float()
        centers = torch.zeros((self.max_people, 2)).float()
        scales = torch.zeros((self.max_people)).float()

        img_hs = np.zeros((self.max_people), dtype=np.float32)
        img_ws = np.zeros((self.max_people), dtype=np.float32)
        focal_lengthes = np.ones((self.max_people), dtype=np.float32)

        # Load image
        imgname = os.path.join(self.dataset_dir, self.imnames[index])
        try:
            origin_img = cv2.imread(imgname)
            origin_img = origin_img[:,:,::-1].copy().astype(np.float32)
        except TypeError:
            print(imgname)
        orig_shape = np.array(origin_img.shape)[:2]
        img_h, img_w = orig_shape

        for idx in range(num_people):
            if idx >= self.max_people:
                break
            valid[idx] = 1.

            bbox = self.bboxs[index][idx].copy().reshape(-1,)
            center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
            scale = 1.0*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200.

            # Process image
            img, crop_ul, crop_br, new_shape, new_x, new_y, old_x, old_y = self.rgb_processing(origin_img, center, sc*scale, rot, flip, pn)

            img = torch.from_numpy(img).float()

            focal_length = self.intris[index][idx].copy()[0][0]

            keypoints = self.pose2ds[index][idx].copy().astype(np.float32)

            if self.dataset_name in self.joint_dataset:
                joints = torch.from_numpy(self.joints[index][idx].copy()).float()
                if joints.shape[1] == 3:
                    conf = (torch.abs(torch.sum(joints, dim=1)) > 0).float().reshape(-1,1)
                    joints = torch.cat([joints, conf], dim=1)
                pose = torch.zeros((72,), dtype=self.dtype)
                betas = torch.zeros((10,), dtype=self.dtype)
                trans = torch.zeros((3,), dtype=self.dtype)
                verts = torch.zeros((6890,3), dtype=self.dtype)
                has_smpl = np.zeros(1)
            else:
                pose = self.poses[index][idx].copy().reshape(72,)
                betas = self.shapes[index][idx].copy().reshape(10,)

                pose = torch.from_numpy(self.pose_processing(pose, rot, flip)).float()
                betas = torch.from_numpy(betas).float()

                temp_pose = pose.clone().reshape(-1, 72)
                temp_shape = betas.clone().reshape(-1, 10)
                temp_trans = torch.zeros((temp_pose.shape[0], 3), dtype=temp_pose.dtype, device=temp_pose.device)
                verts, joints = self.smpl(temp_shape, temp_pose, temp_trans, halpe=True)
                verts = verts.squeeze(0)
                joints = joints.squeeze(0)

                trans = self.estimate_trans_cliff(joints, keypoints, center, focal_length, img_h, img_w)
                trans = torch.from_numpy(trans).float()
            
                conf = torch.ones((len(joints), 1)).float()
                joints = torch.cat([joints, conf], dim=1)

                has_smpl = np.ones(1)

            keypoints[:,:2] = (keypoints[:,:2] - center) / 256
            keypoints = torch.from_numpy(keypoints).float()
            center = torch.from_numpy(np.array(center)).float()


            has_3d[idx] = 1.
            has_smpls[idx] = has_smpl
            imgs[idx] = self.normalize_img(img)
            vertss[idx] = verts
            gt_joints[idx] = joints
            poses[idx] = pose
            shapes[idx] = betas
            gt_trans[idx] = trans
            imgnames[idx] = imgname
            pose2d_gt[idx] = keypoints

            centers[idx] = center
            scales[idx] = sc*scale

            img_hs[idx] = img_h
            img_ws[idx] = img_w
            focal_lengthes[idx] = focal_length


        load_data['valid'] = valid
        load_data['has_3d'] = has_3d
        load_data['has_smpl'] = has_smpls
        # load_data['features'] = imgs
        load_data['verts'] = vertss
        load_data['gt_joints'] = gt_joints
        load_data['img'] = imgs
        load_data['pose'] = poses
        load_data['betas'] = shapes
        load_data['gt_cam_t'] = gt_trans
        load_data['imgname'] = imgnames
        load_data['keypoints'] = pose2d_gt

        load_data["center"] = centers
        load_data["scale"] = scales
        load_data["img_h"] = img_hs
        load_data["img_w"] = img_ws
        load_data["focal_length"] = focal_lengthes


        # try:
        #     origin_img = cv2.imread(imgname)
        #     img = origin_img[:,:,::-1].copy().astype(np.float32)
        # except TypeError:
        #     print(imgname)

        # # self.vis_img('img', origin_img)
        # for center, scale, valid in zip(load_data["center"].detach().numpy(), load_data["scale"].detach().numpy(), load_data['valid']):
        #     if valid == 1:
        #     # Process image
        #         img = origin_img[:,:,::-1].copy().astype(np.float32)
        #         img, crop_ul, crop_br, new_shape, new_x, new_y, old_x, old_y = self.rgb_processing(img, center, sc*scale, rot, flip, pn)
        #         img = torch.from_numpy(img).float()
        #         origin_img = img_crop2origin(img.clone(), origin_img.copy(), new_shape, new_x, new_y, old_x, old_y)
        #         # self.vis_img('img', origin_img)

        # self.vis_input(origin_img, load_data['pred_keypoints'], load_data['keypoints'], load_data['pose'], load_data['betas'], load_data['gt_cam_t'], load_data['valid'], load_data["new_shape"].detach().numpy(), load_data["new_x"].detach().numpy(), load_data["new_y"].detach().numpy(), load_data["old_x"].detach().numpy(), load_data["old_y"].detach().numpy(), focal_length, img_h, img_w)

        return load_data

    def __getitem__(self, index):
        data = self.create_data(index)
        return data

    def __len__(self):
        return self.len













