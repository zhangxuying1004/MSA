import torch
from torch import optim
from tensorboardX import SummaryWriter
import os
from time import time
from datetime import datetime
import numpy as np
import sys
sys.path.append('/home/zhangxuying/Project/Paper_code/MSA/')

# dataset
from OpenSalicon.utils.config import cfg
# from OpenSalicon.utils.dataloader import get_base_dataloader
# from OpenSalicon.utils.dataloader import get_mode_dataloader
# from OpenSalicon.utils.dataloader import get_topic_dataloader
from OpenSalicon.utils.dataloader import get_pseudo_coco_dataloader

# model
from OpenSalicon.utils import network_utils
from OpenSalicon.core.val import val_net
from OpenSalicon.salicon import Salicon

# evaluate metrics
from OpenSalicon.metrics.loss import MyMSELOSS
from OpenSalicon.metrics.normalized_scanpath_salience import MyNormalizedScanpathSalience
from OpenSalicon.metrics.information_gain import InformationGain
from OpenSalicon.metrics.kl_divergence import KLDivergence


def train_net(cfg):
    np.random.seed(cfg.CONST.RNG_SEED)

    # Set up data loader
    # train_data_loader, val_data_loader = get_base_dataloader()
    # train_data_loader, val_data_loader = get_mode_dataloader(mode='diverse')
    # train_data_loader, val_data_loader = get_topic_dataloader(topic_num=10, topic_index=9)
    train_data_loader, val_data_loader = get_pseudo_coco_dataloader()

    # load model
    model = Salicon()
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    model_solver = optim.Adam(
        model.parameters(),
        lr=cfg.TRAIN.READOUT_LEARNING_RATE,
        betas=cfg.TRAIN.BETAS,
        weight_decay=cfg.TRAIN.WEIGHT_DECAY,
    )

    # # choose loss method
    # criterion = WeightedBCELoss(cfg.NETWORK.LOSS_POS_WEIGHT)
    # criterion = MyBCELOSS()
    criterion = MyMSELOSS()
    # criterion = MyKLDLOSS()
    # criterion = nn.BCELoss()

    # load evaluate metrics
    normalized_scanpath_salience = MyNormalizedScanpathSalience()
    info_gain = InformationGain(cfg.NETWORK.EPS)
    kl_divergence = KLDivergence(cfg.NETWORK.EPS)

    # if exists, load the saved model
    init_epoch = 0
    bestnss = -1
    best_epoch = -1
    if 'WEIGHTS' in cfg.CONST and cfg.TRAIN.RESUME_TRAIN:
        print('[INFO] %s Recovering from %s ...' % (datetime.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        # init_epoch = checkpoint['epoch_idx']
        best_nss = checkpoint['best_nss']
        best_epoch = checkpoint['best_epoch']

        model.load_state_dict(checkpoint['model_state_dict'])
        print(
            '[INFO] %s Recover complete. Current epoch #%d, Best NSS = %.4f at epoch #%d.'
            % (datetime.now(), init_epoch, best_nss, best_epoch)
        )

    # Summary writer for TensorBoard
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', datetime.now().isoformat().replace(':', '-'))
    log_dir = output_dir % 'logs'
    ckpt_dir = output_dir % 'checkpoints'
    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(log_dir, 'test'))

    # Load centerBias
    center_bias = network_utils.get_center_bias((cfg.CONST.IMG_H, cfg.CONST.IMG_W), cfg.DATASET.CENTER_BIAS)

    for epoch in range(cfg.TRAIN.NUM_EPOCHES):
        epoch_start_time = time()

        # process batch average meterics
        batch_time = network_utils.AverageMeter()
        data_time = network_utils.AverageMeter()
        losses = network_utils.AverageMeter()
        nsses = network_utils.AverageMeter()
        igs = network_utils.AverageMeter()
        kls = network_utils.AverageMeter()

        batch_end_time = time()
        n_batches = len(train_data_loader)

        for batch_idx, (sample_names, images, fixation_maps) in enumerate(train_data_loader):
            # break
            # Measure data time
            data_time.update(time() - batch_end_time)

            # Ignore imcomplete batches at the end of each epoch
            n_samples = len(sample_names)
            if not n_samples == cfg.CONST.BATCH_SIZE:
                continue

            # get data from data loader
            images = network_utils.var_or_cuda(images)
            fixation_maps = network_utils.var_or_cuda(fixation_maps)
            saliency_maps = model(images)
            loss = criterion(saliency_maps, fixation_maps)

            # model parameters update
            model_solver.zero_grad()
            loss.backward()
            model_solver.step()

            # calculate evaluate metrics
            nss = normalized_scanpath_salience(saliency_maps, fixation_maps)
            ig = info_gain(saliency_maps, fixation_maps, center_bias)
            kl = kl_divergence(saliency_maps, fixation_maps)

            # Append metric to average metrics
            losses.update(loss.item())
            nsses.update(nss.item())
            igs.update(ig.item())
            kls.update(kl.item())

            # Append metric to TensorBoard
            n_itr = epoch * n_batches + batch_idx
            train_writer.add_scalar('Batch/Loss', loss.item(), n_itr)
            train_writer.add_scalar('Batch/NSS', nss, n_itr)
            train_writer.add_scalar('Batch/IG', ig, n_itr)
            train_writer.add_scalar('Batch/KL', kl, n_itr)

            # tick / tock
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            print(
                '[INFO] %s [Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss = %.4f NSS = %.4f IG = %.4f, KL = %.4f'
                % (datetime.now(), epoch + 1, cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, n_batches, batch_time.val, data_time.val, loss.item(),
                   nss, ig, kl)
            )

        # process epoch average meterics
        # Append epoch metric to TensorBoard
        train_writer.add_scalar('Epoch/Loss', losses.avg, epoch + 1)
        train_writer.add_scalar('Epoch/NSS', nsses.avg, epoch + 1)
        train_writer.add_scalar('Epoch/IG', igs.avg, epoch + 1)
        train_writer.add_scalar('Epoch/KL', kls.avg, epoch + 1)

        epoch_end_time = time()
        print(
            '[INFO] %s Epoch [%d/%d] EpochTime = %.3f (s) Loss = %.4f NSS = %.4f IG = %.4f, KL = %.4f'
            % (datetime.now(), epoch + 1, cfg.TRAIN.NUM_EPOCHES, epoch_end_time - epoch_start_time, losses.avg,
               nsses.avg, igs.avg, kls.avg)
        )

        # Validate the training models
        metrics = val_net(cfg, epoch + 1, output_dir, val_data_loader, val_writer, model)
        nss = metrics['nss']

        # save model parameters to file
        if (epoch + 1) % cfg.TRAIN.SAVE_FREQ == 0:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            network_utils.save_checkpoints(
                cfg, os.path.join(ckpt_dir, 'ckpt-epoch-%04d.pth.tar' % (epoch + 1)),
                epoch + 1, model, model_solver, bestnss, best_epoch
            )
            # update the best nss model
            if nss > bestnss:
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)

                bestnss = nss
                best_epoch = epoch + 1
                network_utils.save_checkpoints(
                    cfg, os.path.join(ckpt_dir, 'best-ckpt.pth.tar'),
                    epoch + 1, model, model_solver, bestnss, best_epoch
                )

        # close tensorBoard
        train_writer.close()
        val_writer.close()


if __name__ == '__main__':
    train_net(cfg)
