# coding=utf8
from __future__ import division

import datetime
import logging
import os
import time
from math import ceil

import numpy as np
import torch
from torch.autograd import Variable


def dt():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def trainlog(logfilepath, head='%(message)s'):
    logger = logging.getLogger('mylogger')
    logging.basicConfig(filename=logfilepath, level=logging.INFO, format=head)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(head)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def train(model,
          epoch_num,
          start_epoch,
          optimizer,
          criterion,
          exp_lr_scheduler,
          data_set,
          data_loader,
          save_dir,
          print_inter=200,
          val_inter=3500,
          ):
    step = -1  # 用于计数所有的batch训练次数
    train_size = ceil(len(data_set['train']) / data_loader['train'].batch_size)
    for epoch in range(start_epoch, epoch_num):
        # train phase
        exp_lr_scheduler.step(epoch)
        model.train(True)  # Set model to training mode，这个设置其实只对BN和dropout有影响，训练集要用，测试集不用

        for batch_cnt, data in enumerate(data_loader['train']):  # 每次输出batch的次数，和batch的数据

            step += 1
            model.train(True)
            # print data
            inputs, labels = data

            inputs = Variable(inputs.cuda())
            labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())  # 没有one-hot，因为之后损失函数计算可以直接传入

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)  # 模型前向传播


            loss = 0
            sum_output = 0
            for i in range(len(outputs)):
                loss += criterion(outputs[i], labels)
                sum_output += outputs[i]
            outputs = sum_output / len(outputs)

            _, preds = torch.max(outputs, 1)  # 得到每行预测结果的最大值对应的元组（每行最大值的list，每行最大值对应的列数list）
            loss.backward()
            optimizer.step()

            # batch loss
            if step % print_inter == 0:  # 这是打印出每print_inter个batch的loss，acc
                _, preds = torch.max(outputs, 1)

                batch_corrects = torch.sum((preds == labels)).item()
                batch_acc = batch_corrects / (labels.size(0))

                logging.info('%s [%d-%d] | batch-loss: %.3f | acc@1: %.3f'
                             % (dt(), epoch, batch_cnt, loss.item(), batch_acc))

            if step % val_inter == 0 or (batch_cnt == train_size - 1 and epoch == epoch_num - 1):  # 一定要注意处理最后一个epoch的最后一个batch
                logging.info('current lr:%s' % exp_lr_scheduler.get_lr())
                # val phase
                model.train(False)  # Set model to evaluate mode

                val_loss = 0
                val_corrects = 0
                val_size = ceil(len(data_set['val']) / data_loader['val'].batch_size)  # 得到整个验证集被分为了多少个batch

                t0 = time.time()

                for batch_cnt_val, data_val in enumerate(data_loader['val']):
                    # print data
                    inputs, labels = data_val

                    inputs = Variable(inputs.cuda())
                    labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())

                    # forward
                    outputs = model(inputs)
                    loss = 0
                    sum_output = 0
                    for i in range(len(outputs)):
                        loss += criterion(outputs[i], labels)
                        sum_output += outputs[i]
                    outputs = sum_output / len(outputs)
                    _, preds = torch.max(outputs, 1)

                    # statistics
                    val_loss += loss.item()  # 转tensor为python数据类型，然后求验证集每个batch的loss累加和
                    batch_corrects = torch.sum((preds == labels)).item()  # 得到预测正确的label数累加和，即整个验证集预测正确数
                    val_corrects += batch_corrects

                val_loss = val_loss / val_size  # 整个验证集损失/验证集batch个数
                val_acc = 1.0 * val_corrects / len(data_set['val'])  # 在整个验证集上的预测准确率

                t1 = time.time()
                since = t1 - t0
                logging.info('--' * 30)
                logging.info('current lr:%s' % exp_lr_scheduler.get_lr())

                logging.info('%s epoch[%d]-val-loss: %.4f ||val-acc@1: %.4f ||time: %d'
                             % (dt(), epoch, val_loss, val_acc, since))

                # save model
                save_path = os.path.join(save_dir,
                                         'weights-%d-%d-[%.4f].pth' % (epoch, batch_cnt, val_acc))
                torch.save(model.state_dict(), save_path)
                logging.info('saved model to %s' % (save_path))
                logging.info('--' * 30)

