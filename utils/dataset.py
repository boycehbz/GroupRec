'''
 @FileName    : base.py
 @EditTime    : 2022-10-04 15:54:18
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
import torch.utils.data as data
from utils.FileLoaders import *
import cv2
from alphapose.alphapose_module import prepare
from cliff.cliff_module import prepare_cliff
import torch
from yolox.data.data_augment import preproc

class dataset(data.Dataset):
    def __init__(self, root, subset, annot, data_type='cliff'):

        self.root = root
        self.annot = annot
        self.data_type = data_type

        self.seq_id, self.frame_id = [], []
        for s_id, seq in enumerate(self.annot):
            for f_id, frame in enumerate(seq):
                self.seq_id.append(s_id)
                self.frame_id.append(f_id)

        self.len = len(self.seq_id)

    def create_data(self, index):
        annot = self.annot[self.seq_id[index]][self.frame_id[index]]

        img = cv2.imread(os.path.join(self.root, annot['img_path']))

        bboxes = []
        for person_id in annot.keys():
            if person_id in ['h_w', 'img_path']:
                continue
            bboxes.append(annot[person_id]['bbox'])
        bboxes = np.array(bboxes).reshape(-1, 4)

        data = prepare(img, bboxes)
        data['seq_id'] = self.seq_id[index]
        data['frame_id'] = self.frame_id[index]

        return data

    def create_cliff(self, index):
        annot = self.annot[self.seq_id[index]][self.frame_id[index]]

        img = cv2.imread(os.path.join(self.root, annot['img_path']))

        bboxes, intris = [], []
        for person_id in annot.keys():
            if person_id in ['h_w', 'img_path']:
                continue
            bboxes.append(annot[person_id]['bbox'])
            if annot[person_id]['intri'] is not None:
                intris.append(annot[person_id]['intri'])
        bboxes = np.array(bboxes).reshape(-1, 4)

        if len(intris) == 0:
            intris = None

        data = prepare_cliff(img, bboxes, intris=intris)
        data['seq_id'] = self.seq_id[index]
        data['frame_id'] = self.frame_id[index]

        return data

    def create_yolox(self, index):
        annot = self.annot[self.seq_id[index]][self.frame_id[index]]

        img = os.path.join(self.root, annot['img_path'])

        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img = img.copy()
            img_info["file_name"] = None

        self.test_size = (800, 1440)
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).float()
        if True:
            img = img.half()  # to FP16

        img_info['seq_id'] = self.seq_id[index]
        img_info['frame_id'] = self.frame_id[index]

        return img, img_info

    def __getitem__(self, index):
        if self.data_type == 'cliff':
            data = self.create_cliff(index)
        elif self.data_type == 'yolox':
            data = self.create_yolox(index)
        else:
            data = self.create_data(index)
        return data

    def __len__(self):
        return self.len



