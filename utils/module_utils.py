'''
 @FileName    : module_utils.py
 @EditTime    : 2023-02-03 17:36:12
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
import numpy as np
import cv2
import random
import torch
import os

def viz_annot(dataset_dir, param, smpl):
    # visulize
    img = cv2.imread(os.path.join(dataset_dir, param['img_path']))
    h = int(img.shape[0] / 100)
    del param['img_path']
    if 'h_w' in param.keys():
        img_h, img_w = param['h_w']
        del param['h_w']
    else:
        img_h, img_w = img.shape[:2]
        
    # render = Renderer(resolution=(img_w, img_h))

    for i in param:
        viz_im = img.copy()
        if param[i]['betas'] is not None:
            if 'extri' in param[i].keys():
                extri = np.array(param[i]['extri'])
            else:
                extri = np.eye(4)
            intri = np.array(param[i]['intri'])
            beta = torch.from_numpy(np.array(param[i]['betas'])).reshape(-1, 10).to(torch.float32)
            pose = torch.from_numpy(np.array(param[i]['pose'])).reshape(-1, 72).to(torch.float32)
            trans = torch.from_numpy(np.array(param[i]['trans'])).reshape(-1, 3).to(torch.float32)
            verts, joints = smpl(beta, pose, trans)
            if param[i]['lsp_joints_2d'] is not None:
                j2d = np.array(param[i]['lsp_joints_2d'])
            elif param[i]['halpe_joints_2d_pred'] is not None:
                j2d = np.array(param[i]['halpe_joints_2d_pred'])#[self.halpe2lsp]
            _, mesh_2d = surface_project(verts.detach().numpy()[0], extri, intri)
            # mesh_3d, mesh_2d, gt_cam_t = self.wp_project(verts.detach().numpy()[0], joints.detach().numpy()[0], j2d, self.smpl.faces, viz_im, fx=1500., fy=1500.)
            for p in mesh_2d:
                viz_im = cv2.circle(viz_im, tuple(p.astype(np.int)), 1, (0,255,255), -1)
        if param[i]['halpe_joints_2d'] is not None:
            gt_joints = np.array(param[i]['halpe_joints_2d']).reshape(-1, 3)[:,:2]
            for p in gt_joints:
                viz_im = cv2.circle(viz_im, tuple(p.astype(np.int)), h, (0,0,255), -1)
                # vis_img('img', viz_im)
        if param[i]['halpe_joints_2d_pred'] is not None:
            alpha_joints = np.array(param[i]['halpe_joints_2d_pred']).reshape(-1,3)[:,:2]
            for p in alpha_joints:
                viz_im = cv2.circle(viz_im, tuple(p.astype(np.int)), h, (0,255,0), -1)

        # if param[i]['halpe_joints_2d_det'] is not None:
        #     alpha_joints = np.array(param[i]['halpe_joints_2d_det']).reshape(-1,3)[:,:2]
        #     for p in alpha_joints:
        #         viz_im = cv2.circle(viz_im, tuple(p.astype(np.int)), h, (255,0,0), -1)

        if param[i]['mask_path'] is not None:
            mask = cv2.imread(os.path.join(dataset_dir, param[i]['mask_path']), 0)
            ratiox = 800/int(mask.shape[0])
            ratioy = 800/int(mask.shape[1])
            if ratiox < ratioy:
                ratio = ratiox
            else:
                ratio = ratioy
            cv2.namedWindow('mask',0)
            cv2.resizeWindow('mask',int(mask.shape[1]*ratio),int(mask.shape[0]*ratio))
            #cv2.moveWindow(name,0,0)
            if mask.max() > 1:
                mask = mask/255.
            cv2.imshow('mask',mask)
        if param[i]['bbox'] is not None:
            viz_im = cv2.rectangle(viz_im, tuple(np.array(param[i]['bbox'][0], dtype=np.int)), tuple(np.array(param[i]['bbox'][1], dtype=np.int)), color=(255,255,0), thickness=5)
        # if param[i]['det_bbox'] is not None:
        #     viz_im = cv2.rectangle(viz_im, tuple(np.array(param[i]['det_bbox'][0], dtype=np.int)), tuple(np.array(param[i]['det_bbox'][1], dtype=np.int)), color=(255,0,255), thickness=5)
        #     pass
        vis_img('img', viz_im)

def loss_2d(joints_pred, joints_gt, joint_idx=None):
    joints_pred = np.array(joints_pred)
    joints_gt = np.array(joints_gt)

    if joint_idx is not None:
        joints_pred = joints_pred[:,joint_idx]
        joints_gt = joints_gt[:,joint_idx]

    conf = joints_gt[...,2]

    error = np.mean(np.linalg.norm((joints_pred[...,:2] - joints_gt[...,:2]), axis=2) * conf, axis=1) 
    error = np.mean(error)

    return error


def surface_project(vertices, exter, intri):
    intri_ = np.insert(intri,3,values=0.,axis=1)
    temp_v = np.insert(vertices,3,values=1.,axis=1).transpose((1,0))
    out_point = np.dot(exter, temp_v)
    mesh_3d = out_point.transpose(1,0)[:,:3]
    dis = out_point[2]
    out_point = (np.dot(intri_, out_point) / dis)[:-1]
    mesh_2d = (out_point.astype(np.int32)).transpose(1,0)
    return mesh_3d, mesh_2d

def vis_img(name, im):
    ratiox = 800/int(im.shape[0])
    ratioy = 800/int(im.shape[1])
    if ratiox < ratioy:
        ratio = ratiox
    else:
        ratio = ratioy

    cv2.namedWindow(name,0)
    cv2.resizeWindow(name,int(im.shape[1]*ratio),int(im.shape[0]*ratio))
    #cv2.moveWindow(name,0,0)
    if im.max() > 1:
        im = im/255.
    cv2.imshow(name,im)
    cv2.waitKey()

def draw_keyp(img, joints, color=None, format='coco17', thickness=3):
    skeletons = {'coco17':[[0,1],[1,3],[0,2],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[11,13],[13,15],[6,12],[12,14],    [14,16],[11,12]],
            'halpe':[[0,1],[1,3],[0,2],[2,4],[5,18],[6,18],[18,17],[5,7],[7,9],[6,8],[8,10],[5,11],[11,13],[13,15],[6,12],[12,14],[14,16],[11,19],[19,12],[18,19],[15,24],[15,20],[20,22],[16,25],[16,21],[21,23]],
            'MHHI':[[0,1],[1,2],[3,4],[4,5],[0,6],[3,6],[6,13],[13,7],[13,10],[7,8],[8,9],[10,11],[11,12]],
            'Simple_SMPL':[[0,1],[1,2],[2,6],[6,3],[3,4],[4,5],[6,7],[7,8],[8,9],[8,10],[10,11],[11,12],[8,13],[13,14],[14,15]],
            'LSP':[[0,1],[1,2],[2,3],[5,4],[4,3],[3,9],[9,8],[8,2],[6,7],[7,8],[9,10],[10,11]],
            }
    colors = {'coco17':[(255,0,0), (255,82,0), (255,165,0), (255,210,0), (255,255,0), (127,255,0), (0,127,0), (0,255,0), (0,210,255), (0,127,255), (0,82,127), (0,210,127), (0,0,127), (0,0,255), (139,0,255), (139,0,127)],
                'halpe':[(255,0,0), (255,82,0), (255,165,0), (255,210,0), (255,255,0), (127,255,0), (0,127,0), (0,255,0), (0,210,255), (0,127,255), (0,82,127), (0,210,127), (0,0,127), (0,0,255), (139,0,255), (139,0,127), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0), ],
                'MHHI':[(255,0,0), (255,82,0), (255,165,0), (255,210,0), (255,255,0), (127,255,0), (0,127,0), (0,255,0), (0,210,255), (0,127,255), (0,82,127), (0,210,127), (0,0,127)],
                'Simple_SMPL':[(255,0,0), (255,82,0), (255,165,0), (255,210,0), (255,255,0), (127,255,0), (0,127,0), (0,255,0), (0,210,255), (0,127,255), (0,82,127), (0,210,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127), (0,0,127)],
                'LSP':[(255,0,0), (255,82,0), (255,165,0), (255,210,0), (255,255,0), (127,255,0), (0,127,0), (0,255,0), (0,210,255), (0,127,255), (0,82,127), (0,210,127)]}

    if joints.shape[1] == 3:
        confidence = joints[:,2]
    else:
        confidence = np.ones((joints.shape[0], 1))
    joints = joints[:,:2].astype(np.int)
    for bone, c in zip(skeletons[format], colors[format]):
        if color is not None:
            c = color
        # c = (0,255,255)
        if confidence[bone[0]] > 0.1 and confidence[bone[1]] > 0.1:
            # pass
            img = cv2.line(img, tuple(joints[bone[0]]), tuple(joints[bone[1]]), c, thickness=int(thickness))
    
    for p in joints:
        img = cv2.circle(img, tuple(p), int(thickness * 5/3), c, -1)
        # vis_img('img', img)
    return img


def seed_worker(worker_seed=7):
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed):
    # Set a constant random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    g = torch.Generator()
    g.manual_seed(seed)
    return g

