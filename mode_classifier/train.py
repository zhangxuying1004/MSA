import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import os
from tensorboardX import SummaryWriter

import sys
sys.path.append('/home/zhangxuying/Project/Paper_code/MSA/')
from mode_classifier.utils.config import Parameters
from mode_classifier.utils.dataset import Sub_COCO
from mode_classifier.classifier import Classifier
from mode_classifier.val import calculate_accuracy


# 保存指定epoch，指定idx的 model
def save_model(model_dir, model, epoch=0, idx=0, final=False):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # check_point = model.module.state_dict()
    check_point = model.state_dict()

    if not final:
        model_path = model_dir + str(epoch) + '_' + str(idx) + '.pkl'
    else:
        model_path = model_dir + 'final_model.pkl'
    torch.save(check_point, model_path)


def main():
    torch.initial_seed()
    params = Parameters()

    print('load dataset')
    train_dataset = Sub_COCO(params, mode='train')
    val_dataset = Sub_COCO(params, mode='val')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=True
    )

    print('build model')
    model = Classifier().cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.6)
    criterion = nn.BCELoss()

    # tensorboardX可视化
    writer = SummaryWriter()

    print('training start')
    model.train()
    global_step = 0
    for epoch in range(params.epochs):

        for idx, (train_x, train_y) in enumerate(train_dataloader):
            train_x, train_y = train_x.cuda(), train_y.cuda()
            train_out = model(train_x)
            # 可视化训练集上每个epoch的loss
            loss = criterion(train_out, train_y.float())
            writer.add_scalar('scalar/loss', loss, global_step)
            # 可视化训练集上每个epoch的accuracy
            correct_num = torch.sum(torch.eq((train_out + 0.5).int(), train_y.int())).float()
            accuracy = correct_num / train_y.size(0)
            writer.add_scalar('scalar/accuracy', accuracy, global_step)

            print('Epoch:{}, idx:{}, loss:{:.6f}, train_accuracy:{:.4f}'.format(epoch, idx, loss.item(), accuracy.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 每训练 100 个 epoch，在验证集上验证一次，并保存模型
            if idx % 100 == 0:
                val_accuracy = calculate_accuracy(model, val_dataset, params.batch_size)

                print('val_accuracy:{:.4f}'.format(val_accuracy.item()))
                with open(params.model_logs_dir + 'cider_log.txt', 'a') as f:
                    print(epoch, idx, loss.item(), accuracy.item(), val_accuracy.item(), file=f)

                save_model(params.model_dir, model, epoch, idx)
            global_step += 1
        scheduler.step(epoch)

    save_model(params.model_dir, model, final=True)

    writer.close()
    print('training finished')


if __name__ == '__main__':
    main()
