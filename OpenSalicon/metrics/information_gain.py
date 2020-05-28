import torch


class InformationGain(object):
    """
    Compute the information gain
    """

    def __init__(self, eps=1e-24):
        self.eps = eps

    def __call__(self, saliency_maps, fixation_maps, baseline_maps):
        # print('saliency:', saliency_maps.size())
        # print('fixation:', fixation_maps.size())
        # print(baseline_maps.size())

        baseline_maps = torch.exp(baseline_maps)
        _saliency_maps = saliency_maps.view(saliency_maps.size(0), -1)
        _baseline_maps = baseline_maps.view(baseline_maps.size(0), -1)

        # calculate the max and min values of saliency maps and center bias
        saliency_maps_max = torch.max(_saliency_maps, dim=1)[0].view(saliency_maps.size(0), 1, 1, 1)
        saliency_maps_min = torch.min(_saliency_maps, dim=1)[0].view(saliency_maps.size(0), 1, 1, 1)
        baseline_maps_max = torch.max(_baseline_maps, dim=1)[0].view(baseline_maps.size(0), 1, 1, 1)
        baseline_maps_min = torch.min(_baseline_maps, dim=1)[0].view(baseline_maps.size(0), 1, 1, 1)

        # reshape max and min values to map's size
        saliency_maps_max = saliency_maps_max.repeat(1, saliency_maps.size(1), saliency_maps.size(2),
                                                     saliency_maps.size(3))
        saliency_maps_min = saliency_maps_min.repeat(1, saliency_maps.size(1), saliency_maps.size(2),
                                                     saliency_maps.size(3))
        baseline_maps_max = baseline_maps_max.repeat(1, baseline_maps.size(1), baseline_maps.size(2),
                                                     baseline_maps.size(3))
        baseline_maps_min = baseline_maps_min.repeat(1, baseline_maps.size(1), baseline_maps.size(2),
                                                     baseline_maps.size(3))

        # normalize and vectorize saliency maps
        saliency_maps = (saliency_maps - saliency_maps_min) / (saliency_maps_max - saliency_maps_min)
        baseline_maps = (baseline_maps - baseline_maps_min) / (baseline_maps_max - baseline_maps_min)

        # turn into distributions
        # There is a bug in original code, I change it
        saliency_maps_sum = torch.sum(saliency_maps.view(saliency_maps.size(0), -1), dim=1).view(-1, 1, 1, 1)
        saliency_maps = saliency_maps / saliency_maps_sum
        baseline_maps = baseline_maps / torch.sum(baseline_maps)
        baseline_maps = baseline_maps.repeat(saliency_maps.size(0), 1, 1, 1)

        # mask = fixation_maps.eq(1)
        mask = fixation_maps.gt(0.5)

        ig = torch.log2(saliency_maps[mask] + self.eps) - torch.log2(baseline_maps[mask] + self.eps)
        return torch.mean(ig)


if __name__ == '__main__':
    batch_size = 100
    center_bias = torch.randn(1, 1, 224, 224)
    saliency_maps = torch.randn(batch_size, 1, 224, 224)
    fixation_maps = torch.randn(batch_size, 1, 224, 224)
    fixation_maps[fixation_maps > 0.5] = 1
    fixation_maps[fixation_maps <= 0.5] = 0
    ig = InformationGain()
    print(ig(saliency_maps, fixation_maps, center_bias))
