# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 09:26:28 2020
@author: LiuDahui
"""


# 这个类的定义过程之前在mnist数据集的过程中已经学得比较明确了，直接照搬流程，确定输入输出即可
# 另外，这个地方的一个难点在于，需要将输入和输出都添加封装进来，因为毕竟是数据集嘛，
# 肯定要既有输入也要有target的

import torch
import glob
import os
import numpy as np

from grasp import Grasps
from image import Image,DepthImage

"""
从本地数据集中提取数据，构造成pytorch数据集class，用于训练数据
"""

class Cornell(torch.utils.data.Dataset):
    # 构造函数，载入cornell grasping dataset
    def __init__(self, file_dir, include_depth=True, include_rgb=True, start=0.0, end=1.0):
        '''
        :功能          : 数据集封装类的初始化函数，功能包括数据集读取，数据集划分，其他参数初始化等
        :参数 file_dir : str,按照官方文档的示例和之前的经验，这里需要读入数据集，所以需要指定数据的存放路径
        :参数 include_depth : bool,是否包含depth图像输入
        :参数 include_rgb   : bool,是否包含rgb图像输入
        :参数 start,end : float,为了方便数据集的拆分，这里定义添加两个边界参数start,end
        :返回 None
        ''' 
        super(Cornell,self).__init__() # 继承父类features

        # 参数传递
        self.include_depth = include_depth
        self.include_rgb = include_rgb

        # 去指定文件夹，载入所有需要类别的数据的路径
        cpos_path = glob.glob(os.path.join(file_dir, "*", "pcd*cpos.txt"))
        cpos_path.sort()

        l = len(cpos_path)
        if l == 0:
            raise FileNotFoundError('没有查找到数据集，请检查路径{}'.format(file_dir))
        
        rgb_path = [filename.replace('cpos.txt','r.png') for filename in cpos_path]
        depth_path = [filename.replace('cpos.txt','d.tiff') for filename in cpos_path]

        # 按照设定的边界参数对数据进行划分并指定为类的属性
        self.cpos_path = cpos_path[int(l*start):int(l*end)]
        self.rgb_path = rgb_path[int(l*start):int(l*end)]
        self.depth_path = depth_path[int(l*start):int(l*end)]

        @staticmethod
        def numpy_to_torch(s):
            '''
            :功能     :将输入的numpy数组转化为torch张量，并指定数据类型，如果数据没有channel维度，就给它加上这个维度
            :参数 s   :numpy ndarray,要转换的数组
            :返回     :tensor,转换后的torch张量
            '''
            if len(s.shape) == 2:
                return torch.from_numpy(np.expand_dims(s, axis=0).astype(np.float32))
            else:
                return torch.from_numpy(s.astype(np.float32))
        
        def get_rgb(self, idx):
            '''
            :功能     :读取返回指定id的rgb图像
            :参数 idx :int,要读取的数据id
            :返回     :ndarray,处理好后的rgb图像
            '''
            rgb_img = Image.from_file(self.rgb_path[idx])
            rgb_img.normalize()

            return rgb_img.img
        
        # 与或却rgb图片类似
        def get_depth(self, idx):
            '''
            :功能     :读取返回指定id的depth图像
            :参数 idx :int,要读取的数据id
            :返回     :ndarray,处理好后的depth图像
            '''
            depth_img = DepthImage.from_file(self.depth_path[idx])
            depth_img.normalize()

            return depth_img.img
        
        def get_grasp(self,idx):
            '''
            :功能       :读取返回指定id的抓取标注参数并将多个抓取框的参数返回融合
            :参数 idx   :int,要读取的数据id
            :返回       :以图片的方式返回定义一个抓取的多个参数，包括中心点，角度，宽度和长度
            '''
            grasp_rectangles = Grasps.load_from_cornell_filles(self.cpos_path[idx])
            pos_img, angle_img, width_img = grasp_rectangles.generate_img(shape = (480,640))

            return pos_img, angle_img, width_img
        
        def __getitem__(self, idx):
            # 载入深度图像
            if self.include_depth:
                depth_img = self.get_depth(idx)
                x = self.numpy_to_torch(depth_img)
            # 载入rgb图像
            if self.include_rgb:
                rgb_img = self.get_rgb(idx)
                #torch是要求channel-first的，检测一下，如果读进来的图片是channel-last就调整一下
                if rgb_img.shape[2] == 3: # 3 channels R G B
                    rgb_img = np.moveaxis(rgb_img,2,0)
                x = self.numpy_to_torch(rgb_img)
            # 如果灰度信息和rgb信息都要的话，就把他们堆到一起构成一个四通道的输入
            if self.include_depth and self.include_rgb:
                x = self.numpy_to_torch(np.concatenate(np.expand_dims(depth_img,axis=0), rgb_img), 0)
            # 载入抓取标注参数
            pos_img,angle_img,width_img = self.get_grasp(idx)
            # 处理一下角度信息，因为这个角度值区间比较大，不怎么好处理，所以用两个三角函数把它映射一下：
            cos_img = self.numpy_to_torch(np.cos(2*angle_img))
            sin_img = self.numpy_to_torch(np.sin(2*angle_img))
            
            pos_img = self.numpy_to_torch(pos_img)
            
            # 限定抓取宽度范围并将其映射到[0,1]
            width_img = np.clip(width_img, 0.0, 150.0)/150.0
            width_img = self.numpy_to_torch(width_img)
            
            return x,(pos_img,cos_img,sin_img,width_img)
        
        #映射类型的数据集，别忘了定义这个函数
        def __len__(self):
            return len(self.graspf)
            




