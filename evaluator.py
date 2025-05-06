import os
import numpy as np
import matplotlib.pyplot as plt

from models.networks import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger



# Decide which device we want to run on
# torch.cuda.current_device()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class CDVIS():

    def __init__(self, args, dataloader, input_size):

        self.dataloader = dataloader
        self.args = args
        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids, input_size=input_size)
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        print(self.device)

        # define some other vars to record the training states
        self.running_metric = ConfuseMatrixMeter(n_class=2)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)


        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0

        self.steps_per_epoch = len(dataloader)

        self.activation1 = {}  # 注意！！！！！！！！！！！！！
        self.activation2 = {}  # 注意！！！！！！！！！！！！！
        self.img_A = None
        self.img_B = None

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
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)

    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading best checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device)

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)

    def get_activation(self, name):  # 注意！！！！！！！！！！！！！
        def hook(model, input, output):
            self.activation1[name] = output[0].detach()
            self.activation2[name] = output[1].detach()

        return hook

    def _forward_pass(self, batch_data_A, batch_data_B):

        img_in1 = batch_data_A.to(self.device)
        img_in2 = batch_data_B.to(self.device)
        self.G_pred, self.lay1, self.lay2, self.lay3, self.lay4 = self.net_G(img_in1, img_in2)

    def eval_models(self, checkpoint_name='best_ckpt.pt'):

        self._load_checkpoint(checkpoint_name)

        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self.is_training = False
        self.net_G.eval()
        # 注册网络层名称
        self.net_G.SpaceCenter_Concentrate_Attention.register_forward_hook(self.get_activation('SpaceCenter_Concentrate_Attention'))  # 注意！！！！！！！！！！！！！

        # Iterate over data.
        for self.batch_id, (batch_data_A, batch_data_B, batch_target) in enumerate(self.dataloader, 0):
            if self.batch_id >= 1:
                break
            self.img_A = batch_data_A
            self.img_B = batch_data_B

            with torch.no_grad():
                self._forward_pass(batch_data_A, batch_data_B)
        return self.activation1, self.activation2, self.img_A, self.img_B  # 注意！！！！！！！！！！！！！

class CDEvaluator():

    def __init__(self, args, dataloader, input_size):

        self.dataloader = dataloader
        self.args = args
        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids, input_size=input_size)
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        print(self.device)

        # define some other vars to record the training states
        self.running_metric = ConfuseMatrixMeter(n_class=2)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)


        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0

        self.steps_per_epoch = len(dataloader)

        self.plt_pre = np.array([0])
        self.plt_gt = np.array([0])

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
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)

    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading best checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device)

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)

    def _visualize_pred(self, target):
        if self.args.data_name == 'bayArea':
            pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
            pred_vis = (((pred * -1).add(1)) * 0.5).add(0.5)*255
            target = (((target * -1).add(1)) * 0.5).add(0.5)*255
            return pred_vis.cpu().numpy(), target.cpu().numpy()
        if self.args.data_name == 'china':
            pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
            pred_vis = pred
            target = target
            return pred_vis.cpu().numpy(), target.cpu().numpy()
        if self.args.data_name == 'santaBarbara':
            pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
            pred_vis = (((pred * -1).add(1)) * 0.5).add(0.5)*255
            target = (((target * -1).add(1)) * 0.5).add(0.5)*255
            return pred_vis.cpu().numpy(), target.cpu().numpy()

    def _update_metric(self, batch_target):
        """
        update metric
        """
        target = batch_target.to(self.device).detach()
        G_pred = self.G_pred.detach()
        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _collect_running_batch_states(self, batch_target):

        running_acc = self._update_metric(batch_target)

        m = len(self.dataloader)

        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' %\
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)

        pred_vis, target = self._visualize_pred(batch_target)
        self.plt_pre = np.append(self.plt_pre, pred_vis)
        self.plt_gt = np.append(self.plt_gt, target)


        # if np.mod(self.batch_id, 100) == 1:
        #     vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
        #     vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))
        #
        #     vis_pred = utils.make_numpy_grid(self._visualize_pred())
        #
        #     vis_gt = utils.make_numpy_grid(self.batch['L'])
        #     vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
        #     vis = np.clip(vis, a_min=0.0, a_max=1.0)
        #     file_name = os.path.join(
        #         self.vis_dir, 'eval_' + str(self.batch_id)+'.jpg')
        #     plt.imsave(file_name, vis)

    def _collect_epoch_states(self):

        scores_dict = self.running_metric.get_scores()

        np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)

        self.epoch_acc = scores_dict['mf1']

        with open(os.path.join(self.checkpoint_dir, '%s.txt' % (self.epoch_acc)),
                  mode='a') as file:
            pass

        message = ''
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write('%s\n' % message)  # save the message

        self.logger.write('\n')

    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, batch_data_A, batch_data_B):

        img_in1 = batch_data_A.to(self.device)
        img_in2 = batch_data_B.to(self.device)
        self.G_pred = self.net_G(img_in1, img_in2)
        # self.G_pred, self.lay1, self.lay2, self.lay3, self.lay4 = self.net_G(img_in1, img_in2)

    def eval_models(self, checkpoint_name='best_ckpt.pt'):

        self._load_checkpoint(checkpoint_name)

        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()

        # Iterate over data.
        for self.batch_id, (batch_data_A, batch_data_B, batch_target) in enumerate(self.dataloader, 0):
            with torch.no_grad():
                self._forward_pass(batch_data_A, batch_data_B)
            self._collect_running_batch_states(batch_target)
        self._collect_epoch_states()
        return self.plt_pre, self.plt_gt
