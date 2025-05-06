import os
import random
import numpy as np
import torch
from scipy.io import loadmat
import torch.utils.data as Data
from torch.utils.data import sampler


# 定义各个数据集的根路径
class DataConfig:
    data_name = ""
    root_dir = ""
    def get_data_dir(self, data_name, root_dir='./dataset/'):
        self.data_name = data_name
        self.root_dir = root_dir
        data_dir = os.path.join(self.root_dir, self.data_name)
        print("data dir:"+data_dir)
        return data_dir


# load mat
def load_mat(dir, DatasetName, label_transform):
    if DatasetName == 'riverDataset':
        label_transform = 'norm'
        data_A_dir = os.path.join(dir, 'river_before.mat')
        data_B_dir = os.path.join(dir, 'river_after.mat')
        data_label_dir = os.path.join(dir, 'groundtruth.mat')
        data_A = loadmat(data_A_dir)['river_before']
        data_B = loadmat(data_B_dir)['river_after']
        label = loadmat(data_label_dir)['lakelabel_v1']
    if DatasetName == 'USA':
        label_transform = 'no_norm'
        data_dir = os.path.join(dir, 'USA_Change_Dataset.mat')
        data_A = loadmat(data_dir)['T1']
        data_B = loadmat(data_dir)['T2']
        label = loadmat(data_dir)['Binary']
        input_size = data_A.shape[2]
    if DatasetName == 'china':
        label_transform = 'no_norm'
        data_dir = os.path.join(dir, 'China_Change_Dataset.mat')
        data_A = loadmat(data_dir)['T1']
        data_B = loadmat(data_dir)['T2']
        label = loadmat(data_dir)['Binary']
        input_size = data_A.shape[2]
    if DatasetName == 'santaBarbara':
        label_transform = 'no_norm'
        data_A_dir = os.path.join(dir, 'mat', 'barbara_2013.mat')
        data_B_dir = os.path.join(dir, 'mat', 'barbara_2014.mat')
        data_label_dir = os.path.join(dir, 'mat', 'barbara_gtChanges.mat')
        data_A = loadmat(data_A_dir)['HypeRvieW']
        data_B = loadmat(data_B_dir)['HypeRvieW']
        label = loadmat(data_label_dir)['HypeRvieW']
        input_size = data_A.shape[2]
    if DatasetName == 'bayArea':
        label_transform = 'no_norm'
        data_A_dir = os.path.join(dir, 'mat', 'Bay_Area_2013.mat')
        data_B_dir = os.path.join(dir, 'mat', 'Bay_Area_2015.mat')
        data_label_dir = os.path.join(dir, 'mat', 'bayArea_gtChanges.mat')
        data_A = loadmat(data_A_dir)['HypeRvieW']
        data_B = loadmat(data_B_dir)['HypeRvieW']
        label = loadmat(data_label_dir)['HypeRvieW']
        input_size = data_A.shape[2]
    return data_A, data_B, label, label_transform, input_size


# 归一化处理高光谱数据集
def normalize_data(data):
    np.set_printoptions(threshold=np.inf)
    input_normalize = np.zeros(data.shape)
    for i in range(data.shape[2]):
        input_max = np.max(data[:, :, i])
        input_min = np.min(data[:, :, i])
        #print(data[:, :, i])
        input_normalize[:, :, i] = (data[:, :, i] - input_min) / (input_max - input_min)
        #print(input_normalize[:, :, i])
    return input_normalize


# 寻找各个label的位置
def choose_data_point(data, num_classes, name):
    number_data = []
    pos_data = {}
    #-------------------------for data position------------------------------
    if name == 'bayArea':
        for i in range(num_classes + 1):
            each_class = []
            each_class = np.argwhere(data == (i+1))
            number_data.append(each_class.shape[0])
            pos_data[i] = each_class

        total_pos_data = pos_data[0]
        for i in range(1, num_classes + 1):
            total_pos_data = np.r_[total_pos_data, pos_data[i]]
        total_pos_data = total_pos_data.astype(int)
    if name == 'santaBarbara':
        for i in range(num_classes + 1):
            each_class = []
            each_class = np.argwhere(data == (i+1))
            number_data.append(each_class.shape[0])
            pos_data[i] = each_class

        total_pos_data = pos_data[0]
        for i in range(1, num_classes + 1):
            total_pos_data = np.r_[total_pos_data, pos_data[i]]
        total_pos_data = total_pos_data.astype(int)
    if name == 'USA':
        for i in range(num_classes + 1):
            each_class = []
            each_class = np.argwhere(data == i)
            number_data.append(each_class.shape[0])
            pos_data[i] = each_class

        total_pos_data = pos_data[0]
        for i in range(1, num_classes + 1):
            total_pos_data = np.r_[total_pos_data, pos_data[i]]
        total_pos_data = total_pos_data.astype(int)
    if name == 'china':
        for i in range(num_classes + 1):
            each_class = []
            each_class = np.argwhere(data == i)
            number_data.append(each_class.shape[0])
            pos_data[i] = each_class

        total_pos_data = pos_data[0]
        for i in range(1, num_classes + 1):
            total_pos_data = np.r_[total_pos_data, pos_data[i]]
        total_pos_data = total_pos_data.astype(int)
    return total_pos_data, number_data


# 边界拓展：镜像
def mirror_hsi(height, width, band, input_normalize, patch=3):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    #中心区域
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    #左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding), i, :] = input_normalize[:, padding-i-1, :]
    #右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    #上边镜像
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    #下边镜像
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi


# 获取patch的图像数据
def gain_neighborhood_pixel(mirror_image, point, i, patch=3):
    x = point[i, 0]
    y = point[i, 1]
    temp_image = mirror_image[x:(x + patch), y:(y + patch), :]
    return temp_image


# 获取band的图像数据
def gain_neighborhood_band(x_train, band, band_patch, patch=3):
    nn = band_patch // 2
    pp = (patch * patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch * patch, band)
    if nn == 0:
        return x_train_reshape
    x_train_band = np.zeros((x_train.shape[0], patch * patch * band_patch, band), dtype=float)
    # 中心区域
    x_train_band[:, nn * patch * patch:(nn + 1) * patch * patch, :] = x_train_reshape

    # 左边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:, i * patch * patch:(i + 1) * patch * patch, :i + 1] = x_train_reshape[:, :, band - i - 1:]
            x_train_band[:, i * patch * patch:(i + 1) * patch * patch, i + 1:] = x_train_reshape[:, :, :band - i - 1]
        else:
            x_train_band[:, i:(i + 1), :(nn - i)] = x_train_reshape[:, 0:1, (band - nn + i):]
            x_train_band[:, i:(i + 1), (nn - i):] = x_train_reshape[:, 0:1, :(band - nn + i)]
    # 右边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:, (nn + i + 1) * patch * patch:(nn + i + 2) * patch * patch, :band - i - 1] = x_train_reshape[
                                                                                                        :, :, i + 1:]
            x_train_band[:, (nn + i + 1) * patch * patch:(nn + i + 2) * patch * patch, band - i - 1:] = x_train_reshape[
                                                                     :, :, :i + 1]
        else:
            x_train_band[:, (nn + 1 + i):(nn + 2 + i), (band - i - 1):] = x_train_reshape[:, 0:1, :(i + 1)]
            x_train_band[:, (nn + 1 + i):(nn + 2 + i), :(band - i - 1)] = x_train_reshape[:, 0:1, (i + 1):]
    return x_train_band


# 汇总数据
def X_data(mirror_image, band, point, patch=3, band_patch=1):
    x_train = np.zeros((point.shape[0], patch, patch, band), dtype=float)
    for i in range(point.shape[0]):
        x_train[i, :, :, :] = gain_neighborhood_pixel(mirror_image, point, i, patch)
    print("x shape = {}, type = {}".format(x_train.shape, x_train.dtype))

    x_band = gain_neighborhood_band(x_train, band, band_patch, patch)
    print("x_band shape = {}, type = {}".format(x_band.shape, x_band.dtype))
    print("**************************************************")
    return x_band


# 标签汇总
def Y_label(number_true, num_classes):
    y_true = []
    for i in range(num_classes+1):
        for j in range(number_true[i]):
            y_true.append(i)
    y_true = np.array(y_true)
    print("y_true: shape = {} ,type = {}".format(y_true.shape, y_true.dtype))
    print("**************************************************")
    return y_true

# plt取得位置信息
def plt_pos(args):
    data_name = args.data_name
    root_dir = args.data_root_dir
    data_dir = DataConfig().get_data_dir(data_name, root_dir)
    data_A, data_B, label, label_transform, input_size = load_mat(data_dir, data_name, 'norm')
    if label_transform == 'norm':
        label = label // 255
    total_pos, number = choose_data_point(label, 1, data_name)
    return total_pos, number, data_A.shape[0], data_A.shape[1]

def data_process(args, dir, DatasetName, label_transform='norm'):
    data_A, data_B, label, label_transform, input_size = load_mat(dir, DatasetName, label_transform)
    if label_transform == 'norm':
        label = label // 255
    data_A = normalize_data(data_A)
    data_B = normalize_data(data_B)
    total_pos, number = choose_data_point(label, 1, DatasetName)
    height, width, band = data_A.shape

    mirror_image_A = mirror_hsi(height, width, band, data_A, patch=args.patches)
    mirror_image_B = mirror_hsi(height, width, band, data_B, patch=args.patches)
    A_band = X_data(mirror_image_A, band, total_pos, patch=args.patches, band_patch=args.band_patches)
    B_band = X_data(mirror_image_B, band, total_pos, patch=args.patches, band_patch=args.band_patches)
    '''
    mirror_image_A = mirror_hsi(height, width, band, data_A, args)
    mirror_image_B = mirror_hsi(height, width, band, data_B, args)
    A_band = X_data(mirror_image_A, band, total_pos, args, band_patch=1)
    B_band = X_data(mirror_image_B, band, total_pos, args, band_patch=1)
    '''
    y = Y_label(number, 1)
    return A_band, B_band, y, number, band, input_size


def cut_change_nochange(data, number):
    no_change = data[:number[0], :, :, :]
    change = data[number[0]:, :, :, :]
    return no_change, change


def trans_change_and_nochange(no_change, change):
    change = torch.from_numpy(change.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    no_change = torch.from_numpy(no_change.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    return no_change, change


def choose_train_val(data_set, split):
    # 假设原始的TensorDataset为dataset，包含n个数据样本
    n = len(data_set)
    n_half = split  # 新的TensorDataset包含一定数量的数据样本
    random_indices = random.sample(range(n), n_half)  # 随机选取n_half个样本的下标
    # new_data1 = [data_set[i] for i in random_indices]  # 根据下标选取对应的数据
    # 使用TensorDataset构造函数创建新的TensorDataset
    # train_dataset = Data.TensorDataset(*new_data1)
    train_dataset = Data.TensorDataset(
        data_set.tensors[0][random_indices],
        data_set.tensors[1][random_indices],
        data_set.tensors[2][random_indices]
    )
    # 找出剩余的数据下标
    remain_indices = list(set(range(n)) - set(random_indices))
    remain_indices = random.sample(remain_indices, (n-n_half)//20)
    # 使用剩余的数据下标来构建另一个新的TensorDataset
    val_dataset = Data.TensorDataset(
        data_set.tensors[0][remain_indices],
        data_set.tensors[1][remain_indices],
        data_set.tensors[2][remain_indices]
    )
    return train_dataset, val_dataset


if __name__ == '__main__':
    data1, data2, label, number, band = data_process(3, './dataset/riverDataset', 'riverDataset')

    A_reshape = data1.reshape(data1.shape[0], 3, 3, band)
    A_no_change, A_change = cut_change_nochange(A_reshape, number)
    B_reshape = data2.reshape(data2.shape[0], 3, 3, band)
    B_no_change, B_change = cut_change_nochange(B_reshape, number)
    y_no_change = label[:number[0]]
    y_change = label[number[0]:]

    A_no_change, A_change = trans_change_and_nochange(A_no_change, A_change)
    B_no_change, B_change = trans_change_and_nochange(B_no_change, B_change)
    y_no_change = torch.from_numpy(y_no_change).type(torch.LongTensor)
    y_change = torch.from_numpy(y_change).type(torch.LongTensor)

    data_set_change = Data.TensorDataset(A_change, B_change, y_change)
    data_set_no_change = Data.TensorDataset(A_no_change, B_no_change, y_no_change)

    # 计算划分数据集的 indices
    indices = list(range(len(data_set_change)))
    split1 = int(0.6 * len(indices))
    # split2 = int(0.8 * len(indices))
    # train_indices, val_indices, test_indices = indices[:split1], indices[split1:split2], indices[split2:]

    train_nochange_dataset, val_nochange_dataset = choose_train_val(data_set_no_change, split1)
    train_change_dataset, val_change_dataset = choose_train_val(data_set_change, split1)
    train_dataset = Data.ConcatDataset([train_nochange_dataset, train_change_dataset])
    val_dataset = Data.ConcatDataset([val_nochange_dataset, val_change_dataset])


