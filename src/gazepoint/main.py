# from scipy.io import loadmat

# from datetime import datetime
import os
import requests
import zipfile
import io
import sys
import traceback

# import matlab.engine
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import torch.optim as optim

from logger_module import init_logger
from dataset_loader import GazeDataset
from model_zoo import GazeNet

cosine_similarity = torch.nn.CosineSimilarity()
mse_distance = torch.nn.MSELoss()
bce_loss = torch.nn.BCELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def download_and_extract_zip(url, extract_to, logger):
    if not os.path.exists(extract_to):
        logger.info(f"Downloading {url}")
        response = requests.get(url)
        if response.status_code == 200:
            zip_file = zipfile.ZipFile(io.BytesIO(response.content))
            logger.info(f"Extracting contents to {extract_to}")
            zip_file.extractall(extract_to)
            logger.info(f"Contents extracted to {extract_to}")
        else:
            logger.info("Failed to download the file.")


def F_loss(direction, predict_heatmap, eye_position, gt_position, gt_heatmap):
    # point loss
    heatmap_loss = bce_loss(predict_heatmap, gt_heatmap)

    # angle loss
    gt_direction = gt_position - eye_position
    middle_angle_loss = torch.mean(1 - cosine_similarity(direction, gt_direction))

    return heatmap_loss, middle_angle_loss


def test(net, test_data_loader, logging):
    net.eval()
    total_loss = []
    total_error = []
    info_list = []
    heatmaps = []

    for data in test_data_loader:
        image, face_image, gaze_field, eye_position, gt_position, gt_heatmap = (
            data["image"],
            data["face_image"],
            data["gaze_field"],
            data["eye_position"],
            data["gt_position"],
            data["gt_heatmap"],
        )
        # image, face_image, gaze_field, eye_position, gt_position, gt_heatmap = map(
        #     lambda x: Variable(x.cuda(), volatile=True),
        #     [image, face_image, gaze_field, eye_position, gt_position, gt_heatmap],
        # )

        image = image.to(device)
        face_image = face_image.to(device)
        gaze_field = gaze_field.to(device)
        eye_position = eye_position.to(device)
        gt_position = gt_position.to(device)
        gt_heatmap = gt_heatmap.to(device)

        direction, predict_heatmap = net([image, face_image, gaze_field, eye_position])

        heatmap_loss, m_angle_loss = F_loss(
            direction, predict_heatmap, eye_position, gt_position, gt_heatmap
        )

        loss = heatmap_loss + m_angle_loss

        total_loss.append([heatmap_loss.item(), m_angle_loss.item(), loss.item()])
        logging.info(
            "loss: %.5lf, %.5lf, %.5lf"
            % (heatmap_loss.data[0], m_angle_loss.data[0], loss.data[0])
        )

        middle_output = direction.cpu().data.numpy()
        final_output = predict_heatmap.cpu().data.numpy()
        target = gt_position.cpu().data.numpy()
        eye_position = eye_position.cpu().data.numpy()
        for m_direction, f_point, gt_point, eye_point in zip(
            middle_output, final_output, target, eye_position
        ):
            f_point = f_point.reshape([224 // 4, 224 // 4])
            heatmaps.append(f_point)

            h_index, w_index = np.unravel_index(f_point.argmax(), f_point.shape)
            f_point = np.array([w_index / 56.0, h_index / 56.0])

            f_error = f_point - gt_point
            f_dist = np.sqrt(f_error[0] ** 2 + f_error[1] ** 2)

            # angle
            f_direction = f_point - eye_point
            gt_direction = gt_point - eye_point

            norm_m = (m_direction[0] ** 2 + m_direction[1] ** 2) ** 0.5
            norm_f = (f_direction[0] ** 2 + f_direction[1] ** 2) ** 0.5
            norm_gt = (gt_direction[0] ** 2 + gt_direction[1] ** 2) ** 0.5

            m_cos_sim = (
                m_direction[0] * gt_direction[0] + m_direction[1] * gt_direction[1]
            ) / (norm_gt * norm_m + 1e-6)
            m_cos_sim = np.maximum(np.minimum(m_cos_sim, 1.0), -1.0)
            m_angle = np.arccos(m_cos_sim) * 180 / np.pi

            f_cos_sim = (
                f_direction[0] * gt_direction[0] + f_direction[1] * gt_direction[1]
            ) / (norm_gt * norm_f + 1e-6)
            f_cos_sim = np.maximum(np.minimum(f_cos_sim, 1.0), -1.0)
            f_angle = np.arccos(f_cos_sim) * 180 / np.pi

            total_error.append([f_dist, m_angle, f_angle])
            info_list.append(list(f_point))

    # info_list = np.array(info_list)
    # np.savez('multi_scale_concat_prediction.npz', info_list=info_list)

    # heatmaps = np.stack(heatmaps)
    # np.savez('multi_scale_concat_heatmaps.npz', heatmaps=heatmaps)

    logging.info("average loss : %s" % str(np.mean(np.array(total_loss), axis=0)))
    logging.info("average error: %s" % str(np.mean(np.array(total_error), axis=0)))

    net.train()
    return


def main(*args):
    logger = init_logger()

    # print(os.listdir(dataset_root))
    # eng = matlab.engine.start_matlab()
    # eng.simple_script(nargout=0)
    # eng.quit()

    if len(sys.argv) > 1:
        dataset_root = sys.argv[1]
        train_annotations = os.path.join(dataset_root, sys.argv[2])
        test_annotations = os.path.join(dataset_root, sys.argv[3])
    else:
        dataset_root = os.path.join(
            "/Users/tonmoy/Library/CloudStorage/OneDrive-IndianaUniversity/Research/Education Project/Data/GazeFollow Dataset"
        )
        train_annotations = os.path.join(dataset_root, "train_annotations.mat")
        test_annotations = os.path.join(dataset_root, "test2_annotations.mat")

    logger.info(f"""Dataset root: {dataset_root}""")
    logger.info(f"""Train annotations: {train_annotations}""")
    logger.info(f"""Test annotations: {test_annotations}""")

    download_and_extract_zip(
        "http://gazefollow.csail.mit.edu/downloads/data.zip", dataset_root, logger
    )

    train_set = GazeDataset(
        logging=logger,
        root_dir=dataset_root,
        mat_file=train_annotations,
        training="train",
    )
    test_set = GazeDataset(
        logging=logger,
        root_dir=dataset_root,
        mat_file=test_annotations,
        training="test",
    )

    train_data_loader = DataLoader(
        train_set, batch_size=32 * 4, shuffle=True, num_workers=8
    )
    test_data_loader = DataLoader(
        test_set, batch_size=32 * 4, shuffle=False, num_workers=8
    )

    logger.info(f"""Using Device: {device}""")
    learning_rate = 0.0001

    net = GazeNet()
    net = DataParallel(net)
    net = net.to(device)
    logger.info(f"""Model: {net}""")

    optimizer_s1 = optim.Adam(
        [
            {"params": net.module.face_net.parameters(), "initial_lr": learning_rate},
            {
                "params": net.module.face_process.parameters(),
                "initial_lr": learning_rate,
            },
            {
                "params": net.module.eye_position_transform.parameters(),
                "initial_lr": learning_rate,
            },
            {"params": net.module.fusion.parameters(), "initial_lr": learning_rate},
        ],
        lr=learning_rate,
        weight_decay=0.0001,
    )
    optimizer_s2 = optim.Adam(
        [{"params": net.module.fpn_net.parameters(), "initial_lr": learning_rate}],
        lr=learning_rate,
        weight_decay=0.0001,
    )
    optimizer_s3 = optim.Adam(
        [{"params": net.parameters(), "initial_lr": learning_rate}],
        lr=learning_rate * 0.1,
        weight_decay=0.0001,
    )
    lr_scheduler_s1 = optim.lr_scheduler.StepLR(
        optimizer_s1, step_size=5, gamma=0.1, last_epoch=-1
    )
    lr_scheduler_s2 = optim.lr_scheduler.StepLR(
        optimizer_s2, step_size=5, gamma=0.1, last_epoch=-1
    )
    lr_scheduler_s3 = optim.lr_scheduler.StepLR(
        optimizer_s3, step_size=5, gamma=0.1, last_epoch=-1
    )

    logger.info(f"""Using Learning Rate: {learning_rate}""")
    logger.info(f"""Optimizer: {optimizer_s1}""")
    logger.info(f"""Optimizer: {optimizer_s2}""")
    logger.info(f"""Optimizer: {optimizer_s3}""")

    epochs = 25

    # epoch = 0
    # while epoch < max_epoch:
    #     if epoch == 0:
    #         lr_scheduler = lr_scheduler_s1
    #         optimizer = optimizer_s1
    #     elif epoch == 7:
    #         lr_scheduler = lr_scheduler_s2
    #         optimizer = optimizer_s2
    #     elif epoch == 15:
    #         lr_scheduler = lr_scheduler_s3
    #         optimizer = optimizer_s3

    #     lr_scheduler.step()

    for epoch in range(epochs):
        print(f"""Epoch: {epoch}""")
        if epoch == 0:
            lr_scheduler = lr_scheduler_s1
            optimizer = optimizer_s1
        elif epoch == 7:
            lr_scheduler = lr_scheduler_s2
            optimizer = optimizer_s2
        elif epoch == 15:
            lr_scheduler = lr_scheduler_s3
            optimizer = optimizer_s3

        # lr_scheduler.step()
        try:
            running_loss = []
            for i, data in tqdm(enumerate(train_data_loader)):
                image, face_image, gaze_field, eye_position, gt_position, gt_heatmap = (
                    data["image"],
                    data["face_image"],
                    data["gaze_field"],
                    data["eye_position"],
                    data["gt_position"],
                    data["gt_heatmap"],
                )
                image = image.to(device)
                face_image = face_image.to(device)
                gaze_field = gaze_field.to(device)
                eye_position = eye_position.to(device)
                gt_position = gt_position.to(device)
                gt_heatmap = gt_heatmap.to(device)

                optimizer.zero_grad()

                direction, predict_heatmap = net(
                    [image, face_image, gaze_field, eye_position]
                )

                heatmap_loss, m_angle_loss = F_loss(
                    direction, predict_heatmap, eye_position, gt_position, gt_heatmap
                )

                if epoch == 0:
                    loss = m_angle_loss
                elif epoch >= 7 and epoch <= 14:
                    loss = heatmap_loss
                else:
                    loss = m_angle_loss + heatmap_loss

                logger.info(
                    "Dataindex:%s, Heatmap Loss:%s, Angle Loss:%s, Total Loss:%s"
                    % (
                        str(i),
                        str(heatmap_loss.item()),
                        str(m_angle_loss.item()),
                        str(loss.item()),
                    )
                )

                loss.backward()
                optimizer.step()

                running_loss.append(
                    [heatmap_loss.item(), m_angle_loss.item(), loss.item()]
                )
                if i % 10 == 9:
                    logger.info(
                        "Dataindex:%s Average Loss:%s Mode:%s LR:%s"
                        % (
                            str(i),
                            str(np.mean(running_loss, axis=0)),
                            "adam",
                            str(lr_scheduler.get_last_lr()),
                        )
                    )
                    running_loss = []

            test(net, test_data_loader, logging=logger)
            lr_scheduler.step()
        except Exception as e:
            logger.error(f"""Error: {traceback.print_exc()}""")
            exit(1)


if __name__ == "__main__":
    main()
