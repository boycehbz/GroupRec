'''
 @FileName    : process.py
 @EditTime    : 2022-09-27 16:18:51
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

import torch
import numpy as np
import cv2
from tqdm import tqdm
import time

def extract_valid(data):
    batch_size, agent_num, d = data['keypoints'].shape[:3]
    valid = data['valid'].reshape(-1,)

    data['center'] = data['center'] #.reshape(batch_size*agent_num, -1)[valid == 1]
    data['scale'] = data['scale'] #.reshape(batch_size*agent_num,)[valid == 1]
    data['img_h'] = data['img_h'] #.reshape(batch_size*agent_num,)[valid == 1]
    data['img_w'] = data['img_w'] #.reshape(batch_size*agent_num,)[valid == 1]
    data['focal_length'] = data['focal_length'] #.reshape(batch_size*agent_num,)[valid == 1]

    data['valid_img_h'] = data['img_h'].reshape(batch_size*agent_num,)[valid == 1]
    data['valid_img_w'] = data['img_w'].reshape(batch_size*agent_num,)[valid == 1]
    data['valid_focal_length'] = data['focal_length'].reshape(batch_size*agent_num,)[valid == 1]
    data['has_3d'] = data['has_3d'].reshape(batch_size*agent_num,1)[valid == 1]
    data['has_smpl'] = data['has_smpl'].reshape(batch_size*agent_num,1)[valid == 1]
    data['verts'] = data['verts'].reshape(batch_size*agent_num, 6890, 3)[valid == 1]
    data['gt_joints'] = data['gt_joints'].reshape(batch_size*agent_num, -1, 4)[valid == 1]
    data['pose'] = data['pose'].reshape(batch_size*agent_num, 72)[valid == 1]
    data['betas'] = data['betas'].reshape(batch_size*agent_num, 10)[valid == 1]
    data['keypoints'] = data['keypoints'].reshape(batch_size*agent_num, 26, 3)[valid == 1]
    data['gt_cam_t'] = data['gt_cam_t'].reshape(batch_size*agent_num, 3)[valid == 1]

    imgname = (np.array(data['imgname']).T).reshape(batch_size*agent_num,)[valid.detach().cpu().numpy() == 1]
    data['imgname'] = imgname.tolist()

    return data

def extract_valid_demo(data):
    batch_size, agent_num, _, _, _ = data['img'].shape
    valid = data['valid'].reshape(-1,)

    data['center'] = data['center']
    data['scale'] = data['scale']
    data['img_h'] = data['img_h']
    data['img_w'] = data['img_w']
    data['focal_length'] = data['focal_length']

    data['valid_focal_length'] = data['focal_length'].reshape(batch_size*agent_num,)[valid == 1]

    return data

def to_device(data, device):
    imnames = {'imgname':data['imgname']} 
    data = {k:v.to(device).float() for k, v in data.items() if k not in ['imgname']}
    data = {**imnames, **data}

    return data

def relation_demo(model, loader, device=torch.device('cpu')):

    print('-' * 10 + 'model demo' + '-' * 10)
    model.model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader), total=len(loader)):
            batchsize = data['img'].shape[0]
            data = to_device(data, device)
            data = extract_valid_demo(data)

            # forward
            pred = model.model(data)

            results = {}
            results.update(imgs=data['imgname'])
            results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
            results.update(focal_length=data['valid_focal_length'].detach().cpu().numpy().astype(np.float32))
            if 'pred_verts' not in pred.keys():
                results.update(pred_joints=pred['pred_joints'].detach().cpu().numpy().astype(np.float32))
                model.save_demo_joint_results(results, i, batchsize)
            else:
                results.update(pred_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
                model.save_demo_results(results, i, batchsize)

