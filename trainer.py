import os
import numpy as np
# import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger, Timer

from models.networks import *
from losses import cross_entropy, CustomLoss
from thop import profile
from thop import clever_format
# from utils import de_norm


def get_scheduler(optimizer, args):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            max_epoch = args.max_epochs
            if epoch // 15 >= 1:
                n = int(epoch/15)
                max_epoch = args.max_epochs + n * 15

            lr_l = 1.0 * (args.max_epochs / max_epoch) - epoch / float(max_epoch + 1)
            return lr_l
        # def lambda_rule(epoch):
        #     lr_l = 1.0 - epoch / float(args.max_epochs + 1)
        #     return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs//8
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


class CDTrainer():
    def __init__(self, args, train_dataloader, val_dataloader, input_size):
        self.train_dataloaders = train_dataloader
        self.val_dataloaders = val_dataloader
        self.input_size = input_size

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids, input_size=input_size)

        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        print(self.device)

        # Learning rate and Beta1 for Adam optimizers
        self.lr = args.lr
        # define optimizers
        self.optimizer_G = optim.SGD(self.net_G.parameters(), lr=self.lr,
                                     momentum=0.9,
                                     weight_decay=5e-4)
        # define lr schedulers
        self.exp_lr_scheduler_G = get_scheduler(self.optimizer_G, args)

        self.running_metric = ConfuseMatrixMeter(n_class=2)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)
        # define timer
        self.timer = Timer()
        self.batch_size = args.batch_size

        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = args.max_epochs

        self.global_step = 0
        self.steps_per_epoch = len(train_dataloader)
        self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

        self.G_pred = None
        self.lay1 = None
        self.lay2 = None
        self.lay3 = None
        self.lay4 = None
        self.amp1 = None
        self.amp2 = None
        self.amp3 = None
        self.amp4 = None
        self.G_pred_pha = None
        self.pred_vis = None
        self.G_loss = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        # define the loss functions
        if args.loss == 'ce':
            self._pxl_loss = cross_entropy
            self._loss_pha = CustomLoss()
        else:
            raise NotImplemented(args.loss)

        self.VAL_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'val_acc.npy')):
            self.VAL_ACC = np.load(os.path.join(self.checkpoint_dir, 'val_acc.npy'))
        self.TRAIN_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'train_acc.npy')):
            self.TRAIN_ACC = np.load(os.path.join(self.checkpoint_dir, 'train_acc.npy'))

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)

    # load最后训练的模型继续训练
    def _load_checkpoint(self, ckpt_name='last_ckpt.pt'):
        if os.path.exists(os.path.join(self.checkpoint_dir, ckpt_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, ckpt_name),
                                    map_location=self.device)
            # update net_G states
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.exp_lr_scheduler_G.load_state_dict(
                checkpoint['exp_lr_scheduler_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

            self.logger.write('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.epoch_to_start, self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            print('training from scratch...')

    # 计算时间
    def _timer_update(self):
        self.global_step = (self.epoch_id-self.epoch_to_start) * self.steps_per_epoch + self.batch_id

        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        est = self.timer.estimated_remaining()
        imps = (self.global_step + 1) * self.batch_size / self.timer.get_stage_elapsed()
        return imps, est

    # *255可视化
    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis

    # 存储checkpoint并保存相关信息
    def _save_checkpoint(self, ckpt_name):
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_acc': self.best_val_acc,
            'best_epoch_id': self.best_epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
        }, os.path.join(self.checkpoint_dir, ckpt_name))

    # 每一个epoch更新lr
    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_G.step()

    # 更新混淆矩阵的参数
    def _update_metric(self, batch_target):
        """
        update metric
        """
        target = batch_target.to(self.device).detach()
        G_pred = self.G_pred.detach()

        G_pred = torch.argmax(G_pred, dim=1)
        # 此处应该没有输出
        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    # 计算每100个batch并输出精度
    def _collect_running_batch_states(self, batch_target):

        running_acc = self._update_metric(batch_target)

        m = len(self.train_dataloaders)
        if self.is_training is False:
            m = len(self.val_dataloaders)

        imps, est = self._timer_update()
        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d][%d,%d], imps: %.2f, est: %.2fh, G_loss: %.5f, running_mf1: %.5f\n' %\
                      (self.is_training, self.epoch_id, self.max_num_epochs-1, self.batch_id, m,
                     imps*self.batch_size, est,
                     self.G_loss.item(), running_acc)
            self.logger.write(message)
        # if np.mod(self.batch_id, 500) == 1:
        #     vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
        #     vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))
        #
        #     vis_pred = utils.make_numpy_grid(self._visualize_pred())
        #
        #     vis_gt = utils.make_numpy_grid(self.batch['L'])
        #     vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
        #     vis = np.clip(vis, a_min=0.0, a_max=1.0)
        #     file_name = os.path.join(
        #         self.vis_dir, 'istrain_'+str(self.is_training)+'_'+
        #                       str(self.epoch_id)+'_'+str(self.batch_id)+'.jpg')
        #     plt.imsave(file_name, vis)

    # 计算每一个epoch并输出精度
    def _collect_epoch_states(self):
        scores = self.running_metric.get_scores()
        self.epoch_acc = scores['mf1']
        self.logger.write('Is_training: %s. Epoch %d / %d, epoch_mF1= %.5f\n' %
              (self.is_training, self.epoch_id, self.max_num_epochs-1, self.epoch_acc))
        message = ''
        for k, v in scores.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write(message+'\n')
        self.logger.write('\n')

    def _update_checkpoints(self):

        # save current model
        self._save_checkpoint(ckpt_name='last_ckpt.pt')
        self.logger.write('Lastest model updated. Epoch_acc=%.4f, Historical_best_acc=%.4f (at epoch %d)\n'
              % (self.epoch_acc, self.best_val_acc, self.best_epoch_id))
        self.logger.write('\n')

        # update the best model (based on eval acc)
        if self.epoch_acc > self.best_val_acc:
            self.best_val_acc = self.epoch_acc
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('*' * 10 + 'Best model updated!\n')
            self.logger.write('\n')

    def _update_training_acc_curve(self):
        # update train acc curve
        self.TRAIN_ACC = np.append(self.TRAIN_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'train_acc.npy'), self.TRAIN_ACC)

    def _update_val_acc_curve(self):
        # update val acc curve
        self.VAL_ACC = np.append(self.VAL_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'val_acc.npy'), self.VAL_ACC)

    # 每次epoch开始时清除混淆矩阵
    def _clear_cache(self):
        self.running_metric.clear()

    # 前向 处理输入并得到输出
    def _forward_pass(self, batch_data_A, batch_data_B):
        # self.batch = batch
        img_in1 = batch_data_A.to(self.device)
        img_in2 = batch_data_B.to(self.device)
        self.G_pred = self.net_G(img_in1, img_in2)
        # self.G_pred, self.lay1, self.lay2, self.lay3, self.lay4 = self.net_G(img_in1, img_in2)

    # 反向传播
    def _backward_G(self, batch_target):
        gt = batch_target.to(self.device).long()

        self.G_loss = self._pxl_loss(self.G_pred, gt)
        # with torch.autograd.detect_anomaly():
        #    self.G_loss.backward()
        # self.G_loss = self._pxl_loss(self.G_pred, gt) + self._loss_pha(self.lay1, gt) + self._loss_pha(self.lay2, gt) +\
        #               self._loss_pha(self.lay3, gt) + self._loss_pha(self.lay4, gt)
        self.G_loss.backward()

    def train_models(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        input1 = torch.randn(8,self.input_size,9,9).to(device)
        input2 = torch.randn(8,self.input_size,9,9).to(device)
        flops, params = profile(self.net_G, inputs=(input1, input2))
        flops, params = clever_format([flops, params], '%.3f')
        print(f"运算量：{flops}, 参数量：{params}")
        # self._load_checkpoint()
        # # loop over the dataset multiple times
        # for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):
        #     ################## train #################
        #     ##########################################
        #     self._clear_cache()
        #     self.is_training = True
        #     self.net_G.train()  # Set model to training mode
        #     # Iterate over data.
        #     self.logger.write('lr: %0.7f\n' % self.optimizer_G.param_groups[0]['lr'])
        #     for self.batch_id, (batch_data_A, batch_data_B, batch_target) in enumerate(self.train_dataloaders, 0):
        #         self._forward_pass(batch_data_A, batch_data_B)
        #         # update G
        #         self.optimizer_G.zero_grad()
        #         self._backward_G(batch_target)
        #         self.optimizer_G.step()
        #         self._collect_running_batch_states(batch_target)
        #         self._timer_update()

        #     self._collect_epoch_states()
        #     self._update_training_acc_curve()
        #     self._update_lr_schedulers()


        #     ################## Eval ##################
        #     ##########################################
        #     self.logger.write('Begin evaluation...\n')
        #     self._clear_cache()
        #     self.is_training = False
        #     self.net_G.eval()

        #     # Iterate over data.
        #     for self.batch_id, (batch_data_A, batch_data_B, batch_target) in enumerate(self.val_dataloaders, 0):
        #         with torch.no_grad():
        #             self._forward_pass(batch_data_A, batch_data_B)
        #         self._collect_running_batch_states(batch_target)
        #     self._collect_epoch_states()

        #     ########### Update_Checkpoints ###########
        #     ##########################################
        #     self._update_val_acc_curve()
        #     self._update_checkpoints()

