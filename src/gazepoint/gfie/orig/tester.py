import torch
import torch.nn as nn
import numpy as np
from utils.utils import (
    AverageMeter,
    MovingAverageMeter,
    euclid_dist,
    visualized,
    cosine_sim,
    auc,
)
from tqdm import tqdm

from utils.infer_engine import strategy3dGazeFollowing


class Tester(object):

    def __init__(self, model, criterion, testloader, opt, writer=None):

        self.model = model
        self.criterion = criterion

        self.testloader = testloader

        self.test_dist = AverageMeter()
        self.test_cosine = AverageMeter()
        self.test_dist3d = AverageMeter()

        self.device = torch.device(opt.OTHER.device)

        self.opt = opt
        self.writer = writer

    @torch.no_grad()
    def test(self, epoch, opt):

        self.model.eval()

        self.test_dist.reset()
        self.test_cosine.reset()
        self.test_dist3d.reset()

        loader_capacity = len(self.testloader)
        pbar = tqdm(total=loader_capacity)

        eval_L2dist_counter = AverageMeter()
        eval_3Ddist_counter = AverageMeter()
        eval_AngleError_counter = AverageMeter()
        eval_AUC_counter = AverageMeter()

        for i, data in enumerate(self.testloader, 0):
            x_simg, x_himg, x_hc = data["simg"], data["himg"], data["headloc"]

            x_matrixT = data["matrixT"]

            gaze_vector = data["gaze_vector"]
            gaze_target2d = data["gaze_target2d"]
            gaze_target3d = data["gaze_target3d"].detach().cpu().numpy()

            x_simg = x_simg.to(self.device)
            x_himg = x_himg.to(self.device)
            x_hc = x_hc.to(self.device)
            x_matrixT = x_matrixT.to(self.device)

            inputs_size = x_simg.size(0)

            outs = self.model(x_simg, x_himg, x_hc, x_matrixT)

            pred_heatmap = outs["pred_heatmap"]
            pred_heatmap = pred_heatmap.squeeze(1)
            pred_heatmap = pred_heatmap.data.cpu().numpy()

            pred_gazevector = outs["pred_gazevector"]
            pred_gazevector = pred_gazevector.data.cpu().numpy()

            gaze_vector = gaze_vector.numpy()

            distval = euclid_dist(pred_heatmap, gaze_target2d)
            cosineval = cosine_sim(pred_gazevector, gaze_vector)

            self.test_dist.update(distval, inputs_size)
            self.test_cosine.update(cosineval, inputs_size)

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
            bs = x_simg.size(0)
            for b_idx in range(bs):
                cur_campara = i_campara[b_idx]
                cur_campara = cur_campara.detach().cpu().numpy()
                cur_depmap = i_depmap[b_idx].detach().cpu().numpy()
                cur_pred_gazeheatmap = pred_heatmap[b_idx]
                cur_pred_gazevector = pred_gazevector[b_idx]
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

            pred_gazetarget3d = np.concatenate(pred_gazetarget3d_list, axis=0)
            pred_gazetarget2d = np.concatenate(pred_gazetarget2d_list, axis=0)

            # evaluation
            # print(pred_gazetarget3d, gaze_target3d)
            eval_batch_3Ddist = (
                np.sum(np.linalg.norm(pred_gazetarget3d - gaze_target3d, axis=1)) / bs
            )
            self.test_dist3d.update(eval_batch_3Ddist, inputs_size)
            # eval_batch_l2dist=np.sum(np.linalg.norm(pred_gazetarget2d-gaze_target2d.detach().cpu().numpy(),axis=1))/bs

            # eval_batch_cosine_similarity=np.sum(pred_gazevector*gaze_vector,axis=1)
            # eval_batch_angle_error=np.arccos(eval_batch_cosine_similarity)
            # eval_batch_angle_error=np.sum(np.rad2deg(eval_batch_angle_error))/bs

            # eval_batch_auc=auc(gaze_target2d,pred_heatmap,i_img_size)
            # eval_AUC_counter.update(eval_batch_auc,bs)
            # eval_L2dist_counter.update(eval_batch_l2dist,bs)
            # eval_3Ddist_counter.update(eval_batch_3Ddist,bs)
            # eval_AngleError_counter.update(eval_batch_angle_error,bs)

            pbar.set_postfix(
                dist=self.test_dist.avg,
                cosine=self.test_cosine.avg,
                dist3d=self.test_dist3d.avg,
            )
            pbar.update(1)

        pbar.close()

        if self.writer is not None:
            self.writer.add_scalar(
                "Test dist", self.test_dist.avg, global_step=opt.OTHER.global_step
            )
            self.writer.add_scalar(
                "Test cosine", self.test_cosine.avg, global_step=opt.OTHER.global_step
            )
            self.writer.add_scalar(
                "3D Test dist", self.test_dist3d.avg, global_step=opt.OTHER.global_step
            )
            # self.writer.add_scalar("3D Eval cosine", self.eval_batch_3Ddist.avg, global_step=opt.OTHER.global_step)

        return self.test_dist.avg, self.test_cosine.avg, self.test_dist3d.avg
