import torch


class NormalizedScanpathSalience(object):
    """
    Compute the normalized scanpath salience
    """
    def __call__(self, saliency_maps, fixation_maps):

        mean = torch.mean(saliency_maps)
        std = torch.std(saliency_maps)

        saliency_maps = (saliency_maps - mean) / std
        mask = fixation_maps.eq(1).float()
        n_fixations = torch.sum(mask).item()

        return 1.0 / n_fixations * torch.sum(saliency_maps * mask)


class MyNormalizedScanpathSalience(object):
    """
    Compute the normalized scanpath salience
    """
    def __call__(self, saliency_maps, fixation_maps):

        # print(saliency_maps)
        mean = torch.mean(saliency_maps)
        std = torch.std(saliency_maps)

        # print(mean)
        # print(std)
        saliency_maps = (saliency_maps - mean) / std
        # print(fixation_maps)
        # print(torch.max(fixation_maps))
        # print(torch.min(fixation_maps))
        mask = fixation_maps.gt(0.5).float()
        n_fixations = torch.sum(mask).item()
        # print('n_fixations:', n_fixations)

        return 1.0 / n_fixations * torch.sum(saliency_maps * mask)
