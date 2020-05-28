import torch


class KLDivergence(object):
    def __init__(self, eps=1e-24):
        self.eps = eps

    def __call__(self, saliency_maps, fixation_maps):

        batch_size = saliency_maps.size(0)
        _saliency_maps = saliency_maps.view(saliency_maps.size(0), -1)
        _fixation_maps = fixation_maps.view(fixation_maps.size(0), -1)

        saliency_maps_sum = torch.sum(_saliency_maps, dim=1).view(saliency_maps.size(0), 1, 1, 1)
        fixation_maps_sum = torch.sum(_fixation_maps, dim=1).view(fixation_maps.size(0), 1, 1, 1)
        # print('s m sum:', saliency_maps_sum)
        # print('f m sum:', fixation_maps_sum)
        saliency_maps_sum = saliency_maps_sum.repeat(1, saliency_maps.size(1), saliency_maps.size(2),
                                                     saliency_maps.size(3))
        fixation_maps_sum = fixation_maps_sum.repeat(1, fixation_maps.size(1), fixation_maps.size(2),
                                                     fixation_maps.size(3))

        # make sure saliency_maps and fixation_maps sum to 1
        saliency_maps = saliency_maps / saliency_maps_sum
        fixation_maps = fixation_maps / fixation_maps_sum

        return 1. / batch_size * torch.sum(fixation_maps * torch.log(fixation_maps / (saliency_maps + self.eps) + self.eps))
