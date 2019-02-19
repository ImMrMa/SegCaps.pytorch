import torch.utils.data as data
import os
from skimage import io
import numpy as np
import visdom


class Dataset(data.Dataset):
    def __init__(self, path,transform_data=None,transform_labels=None,data_name='my_ai_000'):
        self.transform_data,self.transform_labels=transform_data,transform_labels
        imgs = []
        files = os.listdir(path)
        for file in files:
            if data_name in file:
                file_path = os.path.join(path, file)
                img = io.imread(file_path)
                imgs.append(img)    
        imgs = np.array(imgs)
        self.labels = self.init_label(imgs)
        self.data=imgs
    def init_label(self, imgs):
        
        flag_leg = imgs == (92, 143, 0)
        flag_body = imgs == (200, 128, 0)
        flag_neck = imgs == (130, 162, 0)
        flag_arm = imgs == (138, 73, 0)
        flag_head = imgs == (0, 236, 163)
        flag_back = imgs == (0, 0, 0)
        flag_leg = flag_leg[:, :, :, 0] * flag_leg[:, :, :, 1] * flag_leg[:, :, :, 2]
        flag_body = flag_body[:, :, :, 0] * flag_body[:, :, :, 1] * flag_body[:, :, :, 2]
        flag_neck = flag_neck[:, :, :, 0] * flag_neck[:, :, :, 1] * flag_neck[:, :, :, 2]
        flag_arm = flag_arm[:, :, :, 0] * flag_arm[:, :, :, 1] * flag_arm[:, :, :, 2]
        flag_head = flag_head[:, :, :, 0] * flag_head[:, :, :, 1] * flag_head[:, :, :, 2]
        flag_back = flag_back[:, :, :, 0] * flag_back[:, :, :, 1] * flag_back[:, :, :, 2]
        flag = np.zeros((imgs.shape[0], imgs.shape[1], imgs.shape[2]), np.long)
        flag[flag_head] = 1
        flag[flag_neck] = 2
        flag[flag_arm] = 3
        flag[flag_body] = 4
        flag[flag_leg] = 5
        flag = np.array(flag)
        return flag

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        if self.transform_data:
            img=self.transform_data(img)
        if self.transform_labels:
            target=self.transform_labels(target)
        return img, target

    def __len__(self):
        return len(self.data)


def test(path):
    a = Dataset(path)
    b, c = a[0]
    vis = visdom.Visdom()
    vis.close(env='test')
    vis.image(
        c.reshape(1, 200, 200).astype(np.float32) / 5,
        env='test',
        win='get'
    )
    for i in range(768):
        vis.image(
            b[i],
            env='test',
            win='tes',
            opts=dict(title='predict')
        )
    print(b.shape)
    print(c.shape)
    print(len(a))

# test('./my_ai/')
