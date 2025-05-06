import torch
import torch.utils.data as Data
import numpy as np
import dataprocess
import os
"""
CD data set with pixel-level labels；
├─GroundTruth.mat
├─river_after.mat
└─river_before.mat
"""


# 按照比例分类还是按照数量分类
def get_loaders(args, data_set_change, data_set_no_change, split):
    train_nochange_dataset, val_nochange_dataset = dataprocess.choose_train_val(data_set_no_change, split)
    train_change_dataset, val_change_dataset = dataprocess.choose_train_val(data_set_change, split)
    train_dataset = Data.ConcatDataset([train_nochange_dataset, train_change_dataset])
    val_dataset = Data.ConcatDataset([val_nochange_dataset, val_change_dataset])
    train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers)
    val_loader = Data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.num_workers)
    return train_loader, val_loader


def get_train_loaders(args):

    data_name = args.data_name
    root_dir = args.data_root_dir
    data_dir = dataprocess.DataConfig().get_data_dir(data_name, root_dir)
    data_AC_dir = os.path.join(data_dir, 'A_change.npy')
    data_ANC_dir = os.path.join(data_dir, 'A_no_change.npy')
    data_BC_dir = os.path.join(data_dir, 'B_change.npy')
    data_BNC_dir = os.path.join(data_dir, 'B_no_change.npy')
    data_YC_dir = os.path.join(data_dir, 'y_change.npy')
    data_YNC_dir = os.path.join(data_dir, 'y_no_change.npy')
    catch = False
    if os.path.exists(data_AC_dir):
    # if catch:
        # LOAD文件
        A_change = np.load(data_AC_dir)
        A_no_change = np.load(data_ANC_dir)
        B_change = np.load(data_BC_dir)
        B_no_change = np.load(data_BNC_dir)
        y_change = np.load(data_YC_dir)
        y_no_change = np.load(data_YNC_dir)
        input_size = A_change.shape[3]
    else:
        A_band, B_band, y, number, band, input_size = dataprocess.data_process(args, data_dir, data_name)

        A_reshape = A_band.reshape(A_band.shape[0], args.patches, args.patches, band)
        A_no_change, A_change = dataprocess.cut_change_nochange(A_reshape, number)
        B_reshape = B_band.reshape(B_band.shape[0], args.patches, args.patches, band)
        B_no_change, B_change = dataprocess.cut_change_nochange(B_reshape, number)
        y_no_change = y[:number[0]]
        y_change = y[number[0]:]
        # 存储文件，简化下次继续
        np.save(data_AC_dir, A_change)
        np.save(data_ANC_dir, A_no_change)
        np.save(data_BC_dir, B_change)
        np.save(data_BNC_dir, B_no_change)
        np.save(data_YC_dir, y_change)
        np.save(data_YNC_dir, y_no_change)

    A_no_change, A_change = dataprocess.trans_change_and_nochange(A_no_change, A_change)
    B_no_change, B_change = dataprocess.trans_change_and_nochange(B_no_change, B_change)
    y_no_change = torch.from_numpy(y_no_change).type(torch.LongTensor)
    y_change = torch.from_numpy(y_change).type(torch.LongTensor)

    data_set_change = Data.TensorDataset(A_change, B_change, y_change)
    data_set_no_change = Data.TensorDataset(A_no_change, B_no_change, y_no_change)

    if args.data_split == False:
        # 计算划分数据集的 indices
        indices = list(range(len(data_set_change)))
        split = int(0.6 * len(indices))
        # split2 = int(0.8 * len(indices))
        # train_indices, val_indices, test_indices = indices[:split1], indices[split1:split2], indices[split2:]
    if args.data_split == True:
        split = int(args.data_split_num)

    train_loader, val_loader = get_loaders(args, data_set_change, data_set_no_change, split)

    return val_loader, train_loader, input_size


def get_test_loaders(args):
    data_name = args.data_name
    root_dir = args.data_root_dir
    data_dir = dataprocess.DataConfig().get_data_dir(data_name, root_dir)
    data_AC_dir = os.path.join(data_dir, 'A_change.npy')
    data_ANC_dir = os.path.join(data_dir, 'A_no_change.npy')
    data_BC_dir = os.path.join(data_dir, 'B_change.npy')
    data_BNC_dir = os.path.join(data_dir, 'B_no_change.npy')
    data_YC_dir = os.path.join(data_dir, 'y_change.npy')
    data_YNC_dir = os.path.join(data_dir, 'y_no_change.npy')
    if os.path.exists(data_AC_dir):
        # LOAD文件
        A_change = np.load(data_AC_dir)
        A_no_change = np.load(data_ANC_dir)
        B_change = np.load(data_BC_dir)
        B_no_change = np.load(data_BNC_dir)
        y_change = np.load(data_YC_dir)
        y_no_change = np.load(data_YNC_dir)
        input_size = A_change.shape[3]

    A_no_change, A_change = dataprocess.trans_change_and_nochange(A_no_change, A_change)
    B_no_change, B_change = dataprocess.trans_change_and_nochange(B_no_change, B_change)
    y_no_change = torch.from_numpy(y_no_change).type(torch.LongTensor)
    y_change = torch.from_numpy(y_change).type(torch.LongTensor)
    data_set_change = Data.TensorDataset(A_change, B_change, y_change)
    data_set_no_change = Data.TensorDataset(A_no_change, B_no_change, y_no_change)
    test_dataset = Data.ConcatDataset([data_set_no_change,data_set_change])
    test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers)

    return test_loader, input_size


def get_vis_loaders(args):
    data_name = args.data_name
    root_dir = args.data_root_dir
    data_dir = dataprocess.DataConfig().get_data_dir(data_name, root_dir)
    data_AC_dir = os.path.join(data_dir, 'A_change.npy')
    data_ANC_dir = os.path.join(data_dir, 'A_no_change.npy')
    data_BC_dir = os.path.join(data_dir, 'B_change.npy')
    data_BNC_dir = os.path.join(data_dir, 'B_no_change.npy')
    data_YC_dir = os.path.join(data_dir, 'y_change.npy')
    data_YNC_dir = os.path.join(data_dir, 'y_no_change.npy')
    if os.path.exists(data_AC_dir):
        # LOAD文件
        A_change = np.load(data_AC_dir)
        A_no_change = np.load(data_ANC_dir)
        B_change = np.load(data_BC_dir)
        B_no_change = np.load(data_BNC_dir)
        y_change = np.load(data_YC_dir)
        y_no_change = np.load(data_YNC_dir)
        input_size = A_change.shape[3]

    A_no_change, A_change = dataprocess.trans_change_and_nochange(A_no_change, A_change)
    B_no_change, B_change = dataprocess.trans_change_and_nochange(B_no_change, B_change)
    y_change = torch.from_numpy(y_change).type(torch.LongTensor)
    data_set_change = Data.TensorDataset(A_change, B_change, y_change)
    vis_loader = Data.DataLoader(data_set_change, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers)

    return vis_loader, input_size


def get_device(args):
    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])
