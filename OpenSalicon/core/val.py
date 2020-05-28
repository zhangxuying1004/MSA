import torch
from utils import network_utils
from datetime import datetime

from metrics.normalized_scanpath_salience import MyNormalizedScanpathSalience
from metrics.information_gain import InformationGain
from metrics.kl_divergence import KLDivergence
from metrics.loss import MyMSELOSS

from utils.network_utils import normalize_map


def val_net(cfg, epoch_idx, output_dir, val_data_loader, val_writer, model):

    # Set up loss functions
    bce_loss = MyMSELOSS()
    # bce_loss = MyBCELOSS()
    # set metrics
    normalized_scanpath_salience = MyNormalizedScanpathSalience()
    info_gain = InformationGain(cfg.NETWORK.EPS)
    kl_divergence = KLDivergence(cfg.NETWORK.EPS)

    # Load CenterBias
    center_bias = network_utils.get_center_bias((cfg.CONST.IMG_H, cfg.CONST.IMG_W), cfg.DATASET.CENTER_BIAS)

    # Testing loop
    n_samples = len(val_data_loader)
    losses = network_utils.AverageMeter()
    nsses = network_utils.AverageMeter()
    igs = network_utils.AverageMeter()
    kls = network_utils.AverageMeter()

    # Switch models to evaluation mode
    model.eval()
    for batch_idx, (sample_name, image, fixation_map) in enumerate(val_data_loader):
        # sample_name = sample_name[0] if isinstance(sample_name[0], str) else sample_name[0].item()
        with torch.no_grad():
            # Get data from data loader
            image = network_utils.var_or_cuda(image)
            fixation_map = network_utils.var_or_cuda(fixation_map)
            saliency_map = model(image)

            loss = bce_loss(saliency_map, fixation_map)

            saliency_map = normalize_map(saliency_map)
            nss = normalized_scanpath_salience(saliency_map, fixation_map)
            ig = info_gain(saliency_map, fixation_map, center_bias)
            kl = kl_divergence(saliency_map, fixation_map)

            # Append loss to average metrics
            losses.update(loss.item())
            nsses.update(nss.item())
            igs.update(ig.item())
            kls.update(kl.item())

            # Add first images to TensorBoard
            if val_writer is not None and batch_idx < 3:
                image = image[0]
                image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
                val_writer.add_image('SaliencyMap#%d/Predicted' % batch_idx, saliency_map[0], epoch_idx)
                val_writer.add_image('SaliencyMap#%d/GroundTruth' % batch_idx, fixation_map[0], epoch_idx)
                val_writer.add_image('SaliencyMap#%d/Origin' % batch_idx, image, epoch_idx)

            # Print sample loss and IoU
            print(
                '[INFO] %s Test[%d/%d] Loss = %.4f NSS = %.4f IG = %.4f, KL = %.4f'
                % (datetime.now(), batch_idx + 1, n_samples, loss.item(), nss, ig, kl)
            )

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Epoch/Loss', losses.avg, epoch_idx)
        val_writer.add_scalar('Epoch/NSS', nsses.avg, epoch_idx)
        val_writer.add_scalar('Epoch/IG', igs.avg, epoch_idx)
        val_writer.add_scalar('Epoch/KL', kls.avg, epoch_idx)

    return {'nss': nsses.avg, 'ig': igs.avg, 'kl': kls.avg}
