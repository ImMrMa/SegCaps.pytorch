import torch.utils.data as data
import lib.dataset as dataset
from torchvision import transforms
import torch
import visdom
import random
import numpy as np
import torch.nn.functional as F


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return torch.from_numpy(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class getcolor():
    def __init__(self):
        color = [i for i in range(5, 256)]
        self.rgb_r = random.sample(color, 251)
        self.rgb_g = random.sample(color, 251)
        self.rgb_b = random.sample(color, 251)
        self.i = 0

    def get(self):
        i = self.i
        color = [self.rgb_r[i % 251], self.rgb_g[i % 251], self.rgb_b[i % 251]]
        idx = random.sample([k for k in range(0, 3)], 3)
        color = [color[idx[0]], color[idx[1]], color[idx[2]]]
        self.i = (i + 1) % 251
        return color


def get_data(batch_size, data_name, data_root='./my_ai/'):
    data_loader = data.DataLoader(
        dataset.Dataset(
            path=data_root,
            transform_data=transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                # channel_change(),

                color_change(),
                ToTensor(),
                tensor_pad(28)
            ]),
            transform_labels=transforms.Compose([
                # transforms.RandomHorizontalFlip(),

                ToTensor(),
                tensor_pad(28)
            ]),
            data_name=data_name
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )
    return data_loader


def test():
    vis = visdom.Visdom()
    vis.close(env='test')
    data_loader = get_data(10)
    a = input('in')
    for batch_index, (data, target) in enumerate(data_loader):
        print(batch_index)
        print(data.shape)
        print(target.shape)
        vis.image(
            data[0, 0, :, :],
            env='test',
            win='image',
            opts=dict(title='target')
        )


class rgb_channel(object):
    def __call__(self, x):
        x = x.reshape(1, -1)
        rgb = torch.zeros(256, x.shape[0]).scatter_(0, x, 1)
        return rgb


class channel_change(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, img):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        color = getcolor()
        flag_leg = img == (92, 143, 0)
        flag_body = img == (200, 128, 0)
        flag_neck = img == (130, 162, 0)
        flag_arm = img == (138, 73, 0)
        flag_head = img == (0, 236, 163)
        flag_back = img == (0, 0, 0)
        flag_leg = flag_leg[:, :, 0] * flag_leg[:, :, 1] * flag_leg[:, :, 2]
        flag_body = flag_body[:, :, 0] * flag_body[:, :, 1] * flag_body[:, :, 2]
        flag_neck = flag_neck[:, :, 0] * flag_neck[:, :, 1] * flag_neck[:, :, 2]
        flag_arm = flag_arm[:, :, 0] * flag_arm[:, :, 1] * flag_arm[:, :, 2]
        flag_head = flag_head[:, :, 0] * flag_head[:, :, 1] * flag_head[:, :, 2]
        flag_back = flag_back[:, :, 0] * flag_back[:, :, 1] * flag_back[:, :, 2]
        img[:, :, :] = color.get()
        img[flag_head] = color.get()
        img[flag_neck] = color.get()
        img[flag_body] = color.get()
        img[flag_leg] = color.get()
        img[flag_arm] = color.get()
        img[flag_back] = [0, 0, 0]
        rgb_r = []
        rgb_g = []
        rgb_b = []
        for i in range(256):
            rgb_r.append(img[:, :, 0, ] == i)
            rgb_g.append(img[:, :, 1] == i)
            rgb_b.append(img[:, :, 2] == i)
        rgb_r, rgb_g, rgb_b = np.array(rgb_r), np.array(rgb_g), np.array(rgb_b)
        # print(rgb_r.shape)
        rgb = np.concatenate((rgb_r, rgb_g, rgb_b), 0)
        # print(rgb.shape)
        data = rgb.astype(np.float32)
        return data * 0.5

    def __repr__(self):
        return self.__class__.__name__ + '()'


class color_change(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, img):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """

        color = getcolor()
        flag_leg = img == (92, 143, 0)
        flag_body = img == (200, 128, 0)
        flag_neck = img == (130, 162, 0)
        flag_arm = img == (138, 73, 0)
        flag_head = img == (0, 236, 163)
        flag_back = img == (0, 0, 0)
        flag_leg = flag_leg[:, :, 0] * flag_leg[:, :, 1] * flag_leg[:, :, 2]
        flag_body = flag_body[:, :, 0] * flag_body[:, :, 1] * flag_body[:, :, 2]
        flag_neck = flag_neck[:, :, 0] * flag_neck[:, :, 1] * flag_neck[:, :, 2]
        flag_arm = flag_arm[:, :, 0] * flag_arm[:, :, 1] * flag_arm[:, :, 2]
        flag_head = flag_head[:, :, 0] * flag_head[:, :, 1] * flag_head[:, :, 2]
        flag_back = flag_back[:, :, 0] * flag_back[:, :, 1] * flag_back[:, :, 2]
        img[:, :, :] = color.get()
        img[flag_head] = color.get()
        img[flag_neck] = color.get()
        img[flag_body] = color.get()
        img[flag_leg] = color.get()
        img[flag_arm] = color.get()
        img[flag_back] = [0, 0, 0]
        img = img.transpose(2, 0, 1)
        return img.astype(np.float32) / 255

    def __repr__(self):
        return self.__class__.__name__ + '()'


class tensor_pad(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self,padding):
        self.padding=padding
    def __call__(self, img):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        img=F.pad(img,(self.padding,self.padding,self.padding,self.padding))
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'

# test()
