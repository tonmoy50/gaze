import torch
from utils.utils import AverageMeter, MovingAverageMeter, euclid_dist, cosine_sim
from utils.infer_engine import strategy3dGazeFollowing
from tqdm import tqdm

import numpy as np


class Trainer(object):
    def __init__(
        self, model, criterion, optimizer, trainloader, valloader, opt, writer=None
    ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.trainloader = trainloader
        self.valloader = valloader

        # for train
        self.losses = MovingAverageMeter()
        self.l2loss = MovingAverageMeter()
        self.l2loss_3d = MovingAverageMeter()
        self.vecloss = MovingAverageMeter()

        self.train_dist = MovingAverageMeter()

        # for eval
        self.eval_dist = AverageMeter()
        self.eval_cosine = AverageMeter()

        self.best_error = None
        self.best_flag = False

        self.device = torch.device(opt.OTHER.device)

        self.opt = opt
        self.writer = writer

    def get_best_error(self, bs_dist, bs_cosine):

        self.best_dist = bs_dist
        self.best_cosine = bs_cosine

    def get_3D_dist_error(self, data, bs, pred_heatmap, pred_gazevector, gaze_target3d):
        # 3D gaze test
        i_depmap = data["depthmap"]
        i_img_size = [
            np.array([i_depmap[i].shape[1], i_depmap[i].shape[0]])[np.newaxis, :]
            for i in range(i_depmap.shape[0])
        ]
        i_img_size = np.concatenate(i_img_size, axis=0)
        i_eye3d = data["eye3d"]
        i_campara = data["campara"]
        pred_gazevector_list = []
        pred_gazetarget2d_list = []
        pred_gazetarget3d_list = []
        # bs = x_simg.size(0)
        for b_idx in range(bs):
            cur_campara = i_campara[b_idx]
            cur_campara = cur_campara.detach().cpu().numpy()
            cur_depmap = i_depmap[b_idx].detach().cpu().numpy()
            cur_pred_gazeheatmap = pred_heatmap[b_idx].detach().cpu().numpy()
            cur_pred_gazevector = pred_gazevector[b_idx].detach().cpu().numpy()
            cur_eye_3d = i_eye3d[b_idx].detach().cpu().numpy()

            pred_result = strategy3dGazeFollowing(
                cur_depmap,
                cur_pred_gazeheatmap,
                cur_pred_gazevector,
                cur_eye_3d,
                cur_campara,
            )

            pred_gazevector_list.append(pred_result["pred_gazevector"])
            pred_gazetarget2d_list.append(pred_result["pred_gazetarget_2d"])
            pred_gazetarget3d_list.append(pred_result["pred_gazetarget_3d"])

        pred_gazetarget3d = torch.tensor(np.concatenate(pred_gazetarget3d_list, axis=0))
        l2_loss = self.criterion[0](pred_gazetarget3d, gaze_target3d)
        l2_loss = torch.mean(l2_loss, dim=1)
        l2_loss = torch.sum(l2_loss) / bs

        return l2_loss

    def train(self, epoch, opt):

        self.model.train()

        # reset loss value
        self.losses.reset()
        self.l2loss.reset()
        self.vecloss.reset()

        self.train_dist.reset()

        self.eval_dist.reset()
        self.eval_cosine.reset()

        loader_capacity = len(self.trainloader)
        pbar = tqdm(total=loader_capacity)
        for i, data in enumerate(self.trainloader, 0):

            self.optimizer.zero_grad()

            opt.OTHER.global_step = opt.OTHER.global_step + 1

            x_simg, x_himg, x_hc = data["simg"], data["himg"], data["headloc"]

            x_matrixT = data["matrixT"]

            gaze_heatmap = data["gaze_heatmap"]
            gaze_vector = data["gaze_vector"]
            gaze_target2d = data["gaze_target2d"]
            gaze_target3d = data["gaze_target3d"]

            x_simg = x_simg.to(self.device)
            x_himg = x_himg.to(self.device)
            x_hc = x_hc.to(self.device)
            x_matrixT = x_matrixT.to(self.device)

            y_gaze_heatmap = gaze_heatmap.to(self.device)
            y_gaze_vector = gaze_vector.to(self.device)

            bs = x_simg.size(0)

            outs = self.model(x_simg, x_himg, x_hc, x_matrixT)

            pred_gheatmap = outs["pred_heatmap"]
            pred_gheatmap = pred_gheatmap.squeeze()

            pred_gvec = outs["pred_gazevector"]
            pred_gvec = pred_gvec.squeeze()

            # gaze heatmap loss
            l2_loss = self.criterion[0](pred_gheatmap, y_gaze_heatmap)
            l2_loss = torch.mean(l2_loss, dim=1)
            l2_loss = torch.mean(l2_loss, dim=1)
            l2_loss = torch.sum(l2_loss) / bs

            # gaze vector loss
            vec_loss = 1 - self.criterion[1](pred_gvec, y_gaze_vector)
            vec_loss = torch.sum(vec_loss) / bs

            l2loss_3d = self.get_3D_dist_error(
                data, bs, pred_gheatmap, pred_gvec, gaze_target3d
            )
            l2loss_scaler_3d = 10

            total_loss = (
                (l2_loss * 10000) + (10 * vec_loss) + (l2loss_3d * l2loss_scaler_3d)
            )
            # total_loss = (l2_loss * 10000) + (10 * vec_loss)
            total_loss.backward()
            self.optimizer.step()

            # record the loss
            self.losses.update(total_loss.item())
            self.l2loss.update(l2_loss.item())
            self.vecloss.update(vec_loss.item())
            self.l2loss_3d.update(l2loss_3d.item())

            # for tensorboardx writer
            if i % opt.OTHER.lossrec_every == 0:

                self.writer.add_scalar(
                    "Train TotalLoss",
                    total_loss.item(),
                    global_step=opt.OTHER.global_step,
                )
                self.writer.add_scalar(
                    "Train L2Loss",
                    l2_loss.item() * 10000,
                    global_step=opt.OTHER.global_step,
                )
                self.writer.add_scalar(
                    "Train l2Loss 3D",
                    l2loss_3d.item() * l2loss_scaler_3d,
                    global_step=opt.OTHER.global_step,
                )
                self.writer.add_scalar(
                    "Train CosLoss",
                    vec_loss.item() * 10,
                    global_step=opt.OTHER.global_step,
                )

                pred_gheatmap = pred_gheatmap.squeeze(1)
                pred_gheatmap = pred_gheatmap.data.cpu().numpy()
                distrain_avg = euclid_dist(pred_gheatmap, gaze_target2d)
                self.train_dist.update(distrain_avg)

            # eval in train procedure on valid dataset
            if (i % opt.OTHER.evalrec_every == 0 and i > 0) or i == (
                loader_capacity - 1
            ):

                self.valid()

                # record L2 distance between predicted 2d gaze target adn GT
                self.writer.add_scalar(
                    "Eval dist", self.eval_dist.avg, global_step=opt.OTHER.global_step
                )
                # record the similarity between predicted gaze vectors and GT
                self.writer.add_scalar(
                    "Eval cosine",
                    self.eval_cosine.avg,
                    global_step=opt.OTHER.global_step,
                )

                self.best_flag = False
                if i == (loader_capacity - 1):
                    if self.best_dist > self.eval_dist.avg:
                        self.best_dist = self.eval_dist.avg
                        self.best_flag = True

                    if self.best_cosine > self.eval_cosine.avg:
                        self.best_cosine = self.eval_cosine.avg
                        self.best_flag = True

            # for tqdm show
            pbar.set_description("Epoch: [{0}]".format(epoch))
            pbar.set_postfix(
                eval_dist=self.eval_dist.avg,
                eval_cos=self.eval_cosine.avg,
                train_dist=self.train_dist.avg,
                totalloss=self.losses.avg,
                l2loss=self.l2loss.avg,
                vecloss=self.vecloss.avg,
                learning_rate=self.optimizer.param_groups[0]["lr"],
            )

            pbar.update(1)

        pbar.close()

    @torch.no_grad()
    def valid(self):

        self.model.eval()

        self.eval_dist.reset()
        self.eval_cosine.reset()

        for i, data in enumerate(self.valloader, 0):

            x_simg, x_himg, x_hc = data["simg"], data["himg"], data["headloc"]

            x_matrixT = data["matrixT"]

            gaze_vector = data["gaze_vector"]
            gaze_target2d = data["gaze_target2d"]

            x_simg = x_simg.to(self.device)
            x_himg = x_himg.to(self.device)
            x_hc = x_hc.to(self.device)
            x_matrixT = x_matrixT.to(self.device)

            bs = x_simg.size(0)
            outs = self.model(x_simg, x_himg, x_hc, x_matrixT)

            pred_heatmap = outs["pred_heatmap"]
            pred_heatmap = pred_heatmap.squeeze(1)
            pred_heatmap = pred_heatmap.data.cpu().numpy()

            pred_gazevector = outs["pred_gazevector"]
            pred_gazevector = pred_gazevector.data.cpu().numpy()
            gaze_vector = gaze_vector.numpy()

            distval = euclid_dist(pred_heatmap, gaze_target2d)
            cosineval = cosine_sim(pred_gazevector, gaze_vector)

            # eval L2 distance between predicted 2d gaze target adn GT
            self.eval_dist.update(distval, bs)

            # eval the similarity between predicted gaze vectors and GT
            self.eval_cosine.update(cosineval, bs)

        self.model.train()
