import sys

sys.path.append('/home/AndrewHR/competition/BD_SJZP/BD_mutilModelPred/')
import os
import numpy as np
import pandas as pd
from dataset.dataset import dataset, collate_fn
import torch
from torch.nn import CrossEntropyLoss
import torch.utils.data as torchdata
from torchvision import datasets, models, transforms
from torchvision.models import resnet50
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from math import ceil
from torch.nn.functional import softmax
from models.resnext101_32x4d_inner3 import multiscale_resnet

test_transforms = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
mode = "test"  # train

rawdata_root = '/home/AndrewHR/competition/BD_SJZP/BD_mutilModelPred/dataset'
train_pd = pd.read_csv("/home/AndrewHR/competition/BD_SJZP/BD_mutilModelPred/dataset/train.txt", sep=" ",
                     header=None, names=['ImageName', 'label'])

true_test_pb = pd.read_csv("/home/AndrewHR/competition/BD_SJZP/BD_mutilModelPred/dataset/test.txt", sep=" ",
                           header=None, names=['ImageName'])
''' addFakeLabel '''
true_test_pb['label'] = 1

test_pd = true_test_pb if mode == "test" else train_pd
print(test_pd.head())

data_set = {}
data_set['test'] = dataset(imgroot=os.path.join(rawdata_root, mode), anno_pd=test_pd,
                           transforms=test_transforms,
                           )
data_loader = {}
data_loader['test'] = torchdata.DataLoader(data_set['test'], batch_size=1, num_workers=4,
                                           shuffle=False, pin_memory=True, collate_fn=collate_fn)

model_name = 'BD_mutilModelPred-6-1-out'  # 用作存预测结果的文件名
resume_1 = '/home/AndrewHR/competition/BD_SJZP/BD_mutilModelPred/models/seresnet101_32x4d_3inner_model/weights-24-240-[0.9927].pth'
resume_2 = '/home/AndrewHR/competition/BD_SJZP/BD_mutilModelPred/models/seresnet101_32x4d_3inner_model/weights-26-60-[0.9927].pth'
resume_3 = '/home/AndrewHR/competition/BD_SJZP/BD_mutilModelPred/models/seresnet101_32x4d_3inner_model/weights-56-160-[0.9927].pth'
resume_4 = '/home/AndrewHR/competition/BD_SJZP/BD_mutilModelPred/models/seresnet101_32x4d_3inner_model/weights-58-180-[0.9951].pth'
resume_5 = '/home/AndrewHR/competition/BD_SJZP/BD_mutilModelPred/models/seresnet101_32x4d_3inner_model/weights-116-160-[0.9951].pth'
resume_6 = '/home/AndrewHR/competition/BD_SJZP/BD_mutilModelPred/models/seresnet101_32x4d_3inner_model/weights-54-140-[0.9878].pth'
#resume_7 = '/home/AndrewHR/competition/BD_SJZP/BD_mutilModelPred/models/seresnet101_32x4d_3inner_model/weights-122-220-[0.9853].pth'

model_1 = multiscale_resnet(num_class=100)
print('resuming finetune from %s' % resume_1)
model_1.load_state_dict(torch.load(resume_1))
model_1 = model_1.cuda()
model_1.eval()

model_2 = multiscale_resnet(num_class=100)
print('resuming finetune from %s' % resume_2)
model_2.load_state_dict(torch.load(resume_2))
model_2 = model_2.cuda()
model_2.eval()

model_3 = multiscale_resnet(num_class=100)
print('resuming finetune from %s' % resume_3)
model_3.load_state_dict(torch.load(resume_3))
model_3 = model_3.cuda()
model_3.eval()

model_4 = multiscale_resnet(num_class=100)
print('resuming finetune from %s' % resume_4)
model_4.load_state_dict(torch.load(resume_4))
model_4 = model_4.cuda()
model_4.eval()

model_5 = multiscale_resnet(num_class=100)
print('resuming finetune from %s' % resume_5)
model_5.load_state_dict(torch.load(resume_5))
model_5 = model_5.cuda()
model_5.eval()

model_6 = multiscale_resnet(num_class=100)
print('resuming finetune from %s' % resume_6)
model_6.load_state_dict(torch.load(resume_6))
model_6 = model_6.cuda()
model_6.eval()

#model_7 = multiscale_resnet(num_class=100)
#print('resuming finetune from %s' % resume_7)
#model_7.load_state_dict(torch.load(resume_7))
#model_7 = model_7.cuda()
#model_7.eval()

criterion = CrossEntropyLoss()

if not os.path.exists('/home/AndrewHR/competition/BD_SJZP/BD_mutilModelPred/predict/Baidu/csv'):
    os.makedirs('/home/AndrewHR/competition/BD_SJZP/BD_mutilModelPred/predict/Baidu/csv')

test_size = ceil(len(data_set['test']) / data_loader['test'].batch_size)  # 得到整个测试集被分成的batch数
test_preds = np.zeros((len(data_set['test'])), dtype=np.float32)
true_label = np.zeros((len(data_set['test'])), dtype=np.int)
idx = 0
test_loss = 0
test_corrects = 0
for batch_cnt_test, data_test in enumerate(data_loader['test']):
    # print data
    print("{0}/{1}".format(batch_cnt_test, int(test_size)))
    inputs, labels = data_test
    inputs = Variable(inputs.cuda())
    labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())
    #inputs = Variable(inputs)
    #labels = Variable(torch.from_numpy(np.array(labels)).long())
    
    # forward
    outputs_1 = model_1(inputs)
    outputs_2 = model_2(inputs)
    outputs_3 = model_3(inputs)
    outputs_4 = model_4(inputs)
    outputs_5 = model_5(inputs)
    outputs_6 = model_6(inputs)
    #outputs_7 = model_7(inputs)
    y_1 = (outputs_1[0]+outputs_2[0]+outputs_3[0]+outputs_4[0]+outputs_5[0]+outputs_6[0])/6
    y_2 = (outputs_1[1]+outputs_2[1]+outputs_3[1]+outputs_4[1]+outputs_5[1]+outputs_6[1])/6
    y_3 = (outputs_1[2]+outputs_2[2]+outputs_3[2]+outputs_4[2]+outputs_5[2]+outputs_6[2])/6
    y_4 = (outputs_1[3]+outputs_2[3]+outputs_3[3]+outputs_4[3]+outputs_5[3]+outputs_6[3])/6
    outputs = [y_1, y_2, y_3, y_4]
    #outputs = outputs_1

    # statistics
    loss = 0
    sum_output = 0
    for i in range(len(outputs)):
        loss += criterion(outputs[i], labels)
        sum_output += outputs[i]
    outputs = sum_output / len(outputs)
        
    _, preds = torch.max(outputs, 1)  # 得到每行预测结果的最大值对应的元组（每行最大值的list，每行最大值对应的列数list）

    test_loss += loss.item()
    batch_corrects = torch.sum((preds == labels)).item()
    test_corrects += batch_corrects
    test_preds[idx:(idx + labels.size(0))] = preds
    true_label[idx:(idx + labels.size(0))] = labels.data.cpu().numpy()
    # statistics
    idx += labels.size(0)
test_loss = test_loss / test_size
test_acc = 1.0 * test_corrects / len(data_set['test'])
print('test-loss: %.4f ||test-acc@1: %.4f'
      % (test_loss, test_acc))

test_pred = test_pd[['ImageName']].copy()
test_pred['pred_label'] = list(test_preds)
test_pred['pred_label'] = test_pred['pred_label'].apply(lambda x: int(x) + 1)
if mode == 'test':
    test_pred[['ImageName', "pred_label"]].to_csv('/home/AndrewHR/competition/BD_SJZP/BD_mutilModelPred/predict/Baidu/csv/{0}_{1}.csv'.format(model_name, mode), sep=" ",
                                                  header=None, index=False)
    print('Save result to %s' % '/home/AndrewHR/competition/BD_SJZP/BD_mutilModelPred/predict/Baidu/csv/{0}_{1}.csv'.format(model_name, mode))
elif mode == 'train':
    test_pred['true_label'] = list(true_label)
    test_pred['true_label'] = test_pred['true_label'].apply(lambda x: int(x) + 1)
    test_pred[['ImageName', "pred_label", "true_label"]].to_csv('/home/AndrewHR/competition/BD_SJZP/BD_mutilModelPred/predict/Baidu/csv/{0}_{1}.csv'.format(model_name, mode),
                                                                sep=" ",
                                                                header=None, index=False)
    print('Save result to %s' % '/home/AndrewHR/competition/BD_SJZP/BD_mutilModelPred/predict/Baidu/csv/{0}_{1}.csv'.format(model_name, mode))
    print('Have error: %d' % test_pred[test_pred['pred_label']!=test_pred['true_label']].shape[0])
print(test_pred.info())
