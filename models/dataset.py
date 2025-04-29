import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        # 加载相机的参数信息
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    # 分解矩阵,将P分解为内参K和外参Rt
    out = cv.decomposeProjectionMatrix(P)
    K = out[0]  # 内参
    R = out[1]  # 外参旋转矩阵
    t = out[2]  # 外参平移矩阵
    '''
    因为分解计算出的K，k22位置上的值不等于1(理论上是必须是1),而是一个接近1的值(eg:1.3或1.5)
    因此K/k22来保证k22位置为1
    fx 0 0
    0 fy 0
    0  0 1
    '''
    K = K / K[2, 2]  # 内参(4×4)
    '''
    fx 0 0 0
    0 fy 0 0
    0  0 1 0
    0  0 0 1
    '''
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)  # 外参(4×4)
    pose[:3, :3] = R.transpose()  # 转置
    pose[:3, 3] = (t[:3] / t[3])[:, 0]  # 与上面类似,分解计算出的t4接近1,保证t4为理论值1

    return intrinsics, pose


class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("CUDA: ", self.device)
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')  # 数据存放的路径
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')

        # 查看是否包含参数camera_outside_sphere,没有返回true
        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        # 查看是否包含参数scale_mat_scale,没有返回1.1
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        # 加载相机投影矩阵
        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        # 所有图片的路径
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'images/*.png')))
        # 图像数量
        self.n_images = len(self.images_lis)
        # 加载图片数据集,并进行归一化处理
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
        # 所有图片对用的mask的路径0
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        # 加载mask数据集,并进行归一化处理
        self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0

        # world_mat is a projection matrix from world to images 图片坐标系到世界坐标系的变换矩阵4×4
        # 从 camera_dict 中提取一系列以 'world_mat_%d' 为键名的矩阵，将它们转换为 float32 类型的 numpy 数组，并存储在 self.world_mats_np 列表中
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        # 用于坐标系归一化(0~1之间),渲染的场景都位于原点的单位球体内
        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []  # 图像数据集对应的内参
        self.pose_all = []  # 图像数据集的外参

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            '''-----中间的是原始的NeuS代码-----'''
            # 对变换矩阵进行缩放
            P = world_mat @ scale_mat
            P = P[:3, :4]  # 去除最后一层的[0 0 0 1]
            # 从相机投影矩阵中拆分出内参和外参的逆
            intrinsics, pose = load_K_Rt_from_P(None, P)
            '''-----中间的是原始的NeuS代码-----'''

            # intrinsics = np.array([[1280, 0, 960, 0], [0, 1280, 540, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            # pose = world_mat
            #
            # world_mat = world_mat @ scale_mat  # 缩放外参矩阵
            # pose = np.eye(4, dtype=np.float32)  # 建立 pose 矩阵
            # pose[:3, :3] = world_mat[:3, :3].transpose()  # 对外参旋转矩阵求转置
            # pose[:, 3] = world_mat[:, 3]  # 外参位置矩阵不变

            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())
        # 图像数据集
        self.images = torch.from_numpy(self.images_np.astype(np.float32)).to(self.device)  # [n_images, H, W, 3]
        # mask数据集
        self.masks = torch.from_numpy(self.masks_np.astype(np.float32)).to(self.device)  # [n_images, H, W, 3]
        # 内参
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)  # [n_images, 4, 4]
        # 内参的逆
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        # 焦距
        self.focal = self.intrinsics_all[0][0, 0]
        # 外参的逆
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        # 图像尺寸
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        # 图像的像素总数
        self.image_pixels = self.H * self.W

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([1.01, 1.01, 1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        # 逆矩阵×矩阵=>单位矩阵[4,4]
        # [4,4]×[4,1]=>[4,1]
        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        print('Load data: End')

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera. 一个摄影机在世界空间生成随机光线
        函数返回了光线rays穿过图片的rgb值以及对应像素位置的mask标签、rays_o(光心)和rays_v(单位方向向量)
        """
        # 在2D图像上随机选择batch_size个像素点(u,v)
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])

        # 获得像素点(u,v)颜色和mask的数据
        color = self.images[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]  # batch_size, 3

        # 相机坐标系下的方向向量:内参(逆)×像素坐标系
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()  # batch_size, 3

        # 单位方向向量:对方向向量做归一化处理
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # batch_size, 3

        # 世界坐标系下的方向向量：外参(逆)×相机坐标系
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3

        # 世界坐标系下的光心位置(外参的逆对应的平移矩阵t)
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)  # batch_size, 3
        return torch.cat([rays_o.to(self.device), rays_v.to(self.device), color, mask[:, :1]],
                         dim=-1).cuda()  # batch_size, 10

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        # rays_d在rays_d的投影,是为了后续做归一化
        # 计算了光线从原点 rays_o 沿方向 rays_d 在一个单位球（半径为1的球）上的最近和最远的交点位置。
        # near 和 far 是这两个交点的位置，通常用于确定光线与球体相交的范围。
        a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)
        # 向量rays_o(原点到光心)在rays_d(单位方向向量)的投影
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        # mid是rays_o在rays_d的投影的终点(的负数)
        mid = 0.5 * (-b) / a
        # 以mid为中点,设定最近点near和最远点far
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)
