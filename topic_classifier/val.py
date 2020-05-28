import torch
from torch.utils.data import DataLoader


def calculate_accuracy(model, dataset, batch_size):
    model.eval()
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )
    correct_num = 0.
    total_num = 0.
    with torch.no_grad():
        for (x, y) in data_loader:
            x, y = x.cuda(), y.cuda()
            out = model(x)
            # correct_num += torch.sum(torch.eq((out + 0.5).int(), y.int())).float()
            correct_num += torch.sum(y.int().eq(torch.argmax(out, dim=1))).float()
            total_num += y.size(0)

    accuracy = correct_num / total_num
    model.train()
    return accuracy
