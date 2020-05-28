import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import os
from tensorboardX import SummaryWriter
from time import time

import sys
sys.path.append('/home/zhangxuying/Project/Paper_code/MSA/')
from topic_classifier.dataset import Sub_COCO
from topic_classifier.classifier import Classifier
from topic_classifier.utils import Parameters
from topic_classifier.val import calculate_accuracy


def save_model(model_dir, model, epoch=0, flag=False):
    # check_point = model.module.state_dict()
    check_point = model.state_dict()
    file_path = os.path.join(model_dir, str(epoch) + '.pkl')
    torch.save(check_point, file_path)

    if flag:
        best_file_path = os.path.join(model_dir, 'best.pkl')
        torch.save(check_point, best_file_path)


def main():

    torch.initial_seed()
    params = Parameters()

    print('加载数据')
    train_dataset = Sub_COCO(params.dataset_path, params.coco_image_dir, params.topic_num, mode='train')
    val_dataset = Sub_COCO(params.dataset_path, params.coco_image_dir, params.topic_num, mode='val')

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=True
    )

    print('构建模型')
    model = Classifier().cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.6)
    criterion = nn.CrossEntropyLoss()

    # tensorboardX可视化
    writer = SummaryWriter()

    print('开始训练')
    model.train()
    global_step = 0
    best_val_accuracy = 0.
    for epoch in range(params.epochs):
        epoch_start_time = time()
        for idx, (train_x, train_y) in enumerate(train_dataloader):
            batch_start_time = time()
            train_x, train_y = train_x.cuda(), train_y.cuda()
            train_out = model(train_x)
            # print('logits:', train_out.size())
            # print('label:', train_y.size())

            loss = criterion(train_out, train_y)
            writer.add_scalar('scalar/loss', loss, global_step)

            correct_num = torch.sum(train_y.int().eq(torch.argmax(train_out, dim=1))).float()
            accuracy = correct_num / train_y.size(0)
            writer.add_scalar('scalar/accuracy', accuracy, global_step)

            print('epoch:{}, idx:{}, loss:{:.6f}, train_accuracy:{:.4f}'.format(epoch, idx, loss.item(), accuracy.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            batch_end_time = time()
            print('{} epoch {} batch cost {} s'.format(epoch, idx, batch_end_time - batch_start_time))
        # 验证模型
        val_accuracy = calculate_accuracy(model, val_dataset, params.batch_size)
        print('epoch:{}, val_accuracy:{:.4f}'.format(epoch, val_accuracy.item()))
        with open(os.path.join(params.model_logs_dir, 'cider_log_10.txt'), 'a') as f:
            print(epoch, val_accuracy.item(), file=f)
        # 保存模型
        flag = False
        if val_accuracy.item() > best_val_accuracy:
            flag = True
            best_val_accuracy = val_accuracy.item()
        save_model(params.model_dir, model, epoch, flag)
        scheduler.step(epoch)

        epoch_end_time = time()
        print('the {} epoch costs {} s'.format(epoch, epoch_end_time - epoch_start_time))

    writer.close()
    print('训练完成')


def test():
    print('train')
    params = Parameters()
    print(os.path.exists(params.model_path))


if __name__ == '__main__':
    main()
    # test()
