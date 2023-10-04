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
from utils.imutils import get_crop, keyp_crop2origin, surface_projection
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
        self.len = len(self.imnames)

    def human_detection(self,):
        model_dir = R'data/bytetrack_x_mot17.pth.tar'
        thres = 0.23
        predictor = Predictor(model_dir, thres)

        for img in self.imnames:
            results, result_img = predictor.predict(img, viz=False)
            self.annot.append(results)

        # dataset_annot = os.path.join(self.dataset_dir,'annot/test.pkl')

        # params = self.load_pkl(dataset_annot)
        # self.imnames, self.img_size, self.intris, self.bbox = [], [], [], []
        # for seq in params:
        #     if len(seq) < 1:
        #         continue
        #     for i, frame in enumerate(seq):
        #         bbox, intris = [], []
        #         for key in frame.keys():
        #             if key in ['img_path', 'h_w']:
        #                 continue
        #             intris.append(np.array(frame[key]['intri'], dtype=self.np_type))
        #             bbox.append(np.array(frame[key]['bbox'], dtype=self.np_type))

        #         self.img_size.append(frame['h_w'])
        #         self.imnames.append(frame['img_path'])
        #         self.intris.append(intris)
        #         self.bbox.append(bbox)
                    
        # del frame
        # del params

        # self.len = len(self.imnames)


    def vis_input(self, image, keypoints, pose, betas, trans):
        # Show image
        image = image.clone().numpy().transpose(1,2,0)
        image = image[:,:,::-1]
        self.vis_img('img', image)

        # Show keypoints
        keypoints = (keypoints[:,:-1].detach().numpy() + 1.) * constants.IMG_RES * 0.5
        keypoints = keypoints.astype(np.int)
        image = (image*255).astype(np.uint8)
        for k in keypoints:
            image = cv2.circle(image, tuple(k), 3, (0,0,255), -1)
        self.vis_img('keyp', image)

        # Show SMPL
        pose = pose.reshape(-1, 72)
        betas = betas.reshape(-1, 10)
        trans = trans.reshape(-1, 3)
        extri = np.eye(4)
        intri = np.array([[5000.,0,112.],[0,5000.,112.],[0,0,1]])
        verts, joints = self.smpl(betas, pose, trans)
        verts = verts.detach().numpy()[0]
        projs, image = surface_projection(verts, self.smpl.faces, extri, intri, image.copy(), viz=False)
        self.vis_img('smpl', image)

    def bbox_from_detector(self, bbox, rescale=1.1):
        """
        Get center and scale of bounding box from bounding box.
        The expected format is [min_x, min_y, max_x, max_y].
        """
        # center
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
        center = torch.tensor([center_x, center_y])

        # scale
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]
        bbox_size = max(bbox_w * 256 / float(192), bbox_h)
        scale = bbox_size / 200.0
        # adjust bounding box tightness
        scale *= rescale
        return center, scale


    def get_transform(self, center, scale, res, rot=0):
        """Generate transformation matrix."""
        # res: (height, width), (rows, cols)
        crop_aspect_ratio = res[0] / float(res[1])
        h = 200 * scale
        w = h / crop_aspect_ratio
        t = np.zeros((3, 3))
        t[0, 0] = float(res[1]) / w
        t[1, 1] = float(res[0]) / h
        t[0, 2] = res[1] * (-float(center[0]) / w + .5)
        t[1, 2] = res[0] * (-float(center[1]) / h + .5)
        t[2, 2] = 1
        if not rot == 0:
            rot = -rot  # To match direction of rotation from cropping
            rot_mat = np.zeros((3, 3))
            rot_rad = rot * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]
            rot_mat[2, 2] = 1
            # Need to rotate around center
            t_mat = np.eye(3)
            t_mat[0, 2] = -res[1] / 2
            t_mat[1, 2] = -res[0] / 2
            t_inv = t_mat.copy()
            t_inv[:2, 2] *= -1
            t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
        return t


    def transform(self, pt, center, scale, res, invert=0, rot=0):
        """Transform pixel location to different reference."""
        t = self.get_transform(center, scale, res, rot=rot)
        if invert:
            t = np.linalg.inv(t)
        new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
        new_pt = np.dot(t, new_pt)
        return np.array([round(new_pt[0]), round(new_pt[1])], dtype=int) + 1



    def crop(self, img, center, scale, res):
        """
        Crop image according to the supplied bounding box.
        res: [rows, cols]
        """
        # Upper left point
        ul = np.array(self.transform([1, 1], center, scale, res, invert=1)) - 1
        # Bottom right point
        br = np.array(self.transform([res[1] + 1, res[0] + 1], center, scale, res, invert=1)) - 1

        # Padding so that when rotated proper amount of context is included
        pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)

        new_shape = [br[1] - ul[1], br[0] - ul[0]]
        if len(img.shape) > 2:
            new_shape += [img.shape[2]]
        new_img = np.zeros(new_shape, dtype=np.float32)

        # Range to fill new array
        new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
        new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
        # Range to sample from original image
        old_x = max(0, ul[0]), min(len(img[0]), br[0])
        old_y = max(0, ul[1]), min(len(img), br[1])
        try:
            new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]
        except Exception as e:
            print(e)

        new_img = cv2.resize(new_img, (res[1], res[0]))  # (cols, rows)

        return new_img, ul, br


    def process_image(self, orig_img_rgb, bbox,
                    crop_height=256,
                    crop_width=192):
        """
        Read image, do preprocessing and possibly crop it according to the bounding box.
        If there are bounding box annotations, use them to crop the image.
        If no bounding box is specified but openpose detections are available, use them to get the bounding box.
        """
        try:
            center, scale = self.bbox_from_detector(bbox)
        except Exception as e:
            print("Error occurs in person detection", e)
            # Assume that the person is centered in the image
            height = orig_img_rgb.shape[0]
            width = orig_img_rgb.shape[1]
            center = np.array([width // 2, height // 2])
            scale = max(height, width * crop_height / float(crop_width)) / 200.

        img, ul, br = self.crop(orig_img_rgb, center, scale, (crop_height, crop_width))
        crop_img = img.copy()

        img = img / 255.
        mean = np.array(constants.IMG_NORM_MEAN, dtype=np.float32)
        std = np.array(constants.IMG_NORM_STD, dtype=np.float32)
        norm_img = (img - mean) / std
        norm_img = np.transpose(norm_img, (2, 0, 1))

        return norm_img, center, scale, ul, br, crop_img

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
        intris = None
        
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
        focal_lengthes = torch.zeros((batch_size, self.max_people)).float()
        valid = torch.zeros((batch_size, self.max_people)).float()

        for batch_idx in range(batch_size):
            boxes = annot['bbox'][batch_idx * self.max_people:(batch_idx + 1) * self.max_people]
            num_people = len(boxes)
            
            for i in range(len(boxes)):
                if i >= num_people:
                    break
                valid[batch_idx][i] = 1.

                if intris is not None:
                    focal_length = intris[i][0][0]
                else:
                    focal_length = (img_h ** 2 + img_w ** 2) ** 0.5

                bbox = boxes[i]

                norm_img, center, scale, crop_ul, crop_br, _ = self.process_image(img.copy(), bbox.reshape(-1,))

                # Get 2D keypoints and apply augmentation transforms
                h = 200 * scale
                s = float(256) / h

                norm_imgs[i] = torch.from_numpy(norm_img)
                centers[i] = center
                scales[i] = scale
                crop_uls[i] = crop_ul
                crop_brs[i] = crop_br
                img_hs[i] = img_h
                img_ws[i] = img_w
                focal_lengthes[i] = focal_length

            load_data['valid'] = valid
            load_data["norm_img"] = norm_imgs
            load_data["center"] = centers
            load_data["scale"] = scales
            load_data["crop_ul"] = torch.from_numpy(crop_uls)
            load_data["crop_br"] = torch.from_numpy(crop_brs)
            load_data["img_h"] = torch.from_numpy(img_hs)
            load_data["img_w"] = torch.from_numpy(img_ws)
            load_data["focal_length"] = torch.from_numpy(focal_lengthes)


        return load_data

    def __getitem__(self, index):
        data = self.create_data(index)
        return data

    def __len__(self):
        return self.len













