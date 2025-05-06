from argparse import ArgumentParser
from trainer import *
from evaluator import *
import data_loaders
import dataprocess
import torch


"""
the main function for training the CD networks
"""


def train(args):
    val_dataloader, train_dataloader, input_size = data_loaders.get_train_loaders(args)
    model = CDTrainer(args=args, train_dataloader=train_dataloader, val_dataloader=val_dataloader, input_size=input_size)
    model.train_models()


def test(args):
    dataloader, input_size = data_loaders.get_test_loaders(args)
    model = CDEvaluator(args=args, dataloader=dataloader, input_size=input_size)
    pre, gt = model.eval_models()

    pos, number, x_len, y_len = dataprocess.plt_pos(args)
    vis_pre = np.zeros((x_len, y_len), dtype=int)
    vis_gt = np.zeros((x_len, y_len), dtype=int)
    for index, i in enumerate(pos):
        vis_pre[i[0]][i[1]] = pre[index + 1]
    for index, i in enumerate(pos):
        vis_gt[i[0]][i[1]] = gt[index + 1]
    vis_dir = args.vis_dir
    file_name = os.path.join(vis_dir, 'eval_gt' + '.jpg')
    plt.imsave(file_name, vis_gt, cmap='gray')
    file_name = os.path.join(vis_dir, 'eval_pre' + '.jpg')
    plt.imsave(file_name, vis_pre, cmap='gray')

def vis(args):
    vis_dataloader, input_size = data_loaders.get_vis_loaders(args)  # 加载需要可视化的数据样本（部分）
    model = CDVIS(args=args, dataloader=vis_dataloader, input_size=input_size)
    activation1, att, data_A, data_B = model.eval_models()
    act = activation1['SpaceCenter_Concentrate_Attention']  # 注意！！！！！！！！！！
    act1 = att['SpaceCenter_Concentrate_Attention']  # 获得att map
    print(act.size())
    print(act1.size())

    act1 = torch.squeeze(act1)
    act = torch.stack([act[0][91],act[0][103],act[0][123]])

    print(act.size())
    print(act1.size())

    vis_dir = args.vis_dir
    file_name = os.path.join(vis_dir, 'VIS_A' + '.jpg')
    plt.imsave(file_name, np.transpose(torch.stack([data_A[0][91],data_A[0][103],data_A[0][123]],0), (1, 2, 0)).detach().cpu().numpy())
    file_name = os.path.join(vis_dir, 'VIS_B' + '.jpg')
    plt.imsave(file_name, np.transpose(torch.stack([data_B[0][91],data_B[0][103],data_B[0][123]],0), (1, 2, 0)).detach().cpu().numpy())
    file_name = os.path.join(vis_dir, 'act' + '.jpg')
    plt.imsave(file_name, np.transpose(act.detach().cpu().numpy(), (1, 2, 0)))
    file_name = os.path.join(vis_dir, 'att' + '.jpg')
    plt.imsave(file_name, (act1[0]*255).detach().cpu().numpy())




if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='bay_p9_8', type=str)
    parser.add_argument('--checkpoint_root', default='./checkpoints', type=str)
    parser.add_argument('--checkpoint_dir', default='./checkpoints', type=str)
    parser.add_argument('--vis_dir', default='./checkpoints', type=str)
    # data
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_root_dir', default='dataset', type=str)
    parser.add_argument('--data_name', default='china', type=str)
    parser.add_argument('--data_split', default=True, type=bool, help='Is it divided into training set and test set according to quantity')
    parser.add_argument('--data_split_num', default=800, type=int, help='When data_split is true, the number of training sets')

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--patches', type=int, default=9, help='number of patches')
    parser.add_argument('--band_patches', type=int, default=1, help='number of band patches')

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--net_G', default='unet_transformer_u', type=str,
                        help='base_resnet18 | base_transformer_pos_s4 | '
                             'base_transformer_pos_s4_dd8 | '
                             'base_transformer_pos_s4_dd8_dedim8|')
    parser.add_argument('--loss', default='ce', type=str)

    # optimizer
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--max_epochs', default=50, type=int)
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step')
    parser.add_argument('--lr_decay_iters', default=100, type=int)
    args = parser.parse_args()
    data_loaders.get_device(args)
    print(torch.cuda.is_available())
    print(args.gpu_ids)
    seed = 42
    torch.manual_seed(seed)

    #  保存checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    # 创建保存checkpoints的路径
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir可视化路径
    args.vis_dir = os.path.join('vis', args.project_name)
    # 创建保存可视化路径
    os.makedirs(args.vis_dir, exist_ok=True)

    train(args)
    # test(args)
    # vis(args)
