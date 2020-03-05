import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset_g import MyDataset_G
from densenet import densenet121
from torchvision import transforms
import time
import torch.optim as optim
import os
import itertools
from Dnn import Batch_Net

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 10
num_class = 3
num_epochs = 300
path = "/home/xf/ISIC-data2017/"
image_datasets = {x: MyDataset_G(csv_path=path + x + ".csv",
                                 image_path=path + x,
                                 lbp_path=path + x + "_944.csv",
                                 data_transforms=data_transforms,
                                 dataset=x) for x in ['train', 'test']}

dataloders = {x: DataLoader(image_datasets[x],
                            batch_size=batch_size,
                            shuffle=True) for x in ['train', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

model = densenet121(pretrained=True)

model = model.to(device)

model2 = Batch_Net(1944, 16, 8, 3)

model2 = model2.to(device)

criterion = nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.SGD(itertools.chain(model.parameters(), model2.parameters()), lr=0.001, momentum=0.9,
                            weight_decay=0.0005)

since = time.time()

best_acc = 0.0
best0 = 0
best1 = 0
best2 = 0
for epoch in range(num_epochs):
    begin_time = time.time()
    count_batch = 0

    print('Epoch {}/{}'.format(epoch + 1, num_epochs))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'test']:
        if phase == 'train':
            model.train(True)  # Set model to training mode
        else:
            model.train(False)  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0
        counti = 0
        class0 = 0
        class1 = 0
        class2 = 0
        for inputs, inputs2, labels in dataloders[phase]:
            count_batch += 1
            # get the inputs
            inputs = inputs.to(device)
            inputs2 = inputs2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            x = torch.cat((outputs, inputs2), 1)

            y = model2(x)
            _, preds = torch.max(y.data, 1)
            loss = criterion(y, labels)

            if phase == 'train':
                loss.backward()
                optimizer.step()
            running_loss += float(loss.item())
            running_corrects += torch.sum(preds == labels.data)
            counti += inputs.size(0)
            if phase == 'test':
                for k in range(inputs.size(0)):
                    if labels.data[k] == 0 and preds[k] == 0:
                        class0 += 1
                    if labels.data[k] == 1 and preds[k] == 1:
                        class1 += 1
                    if labels.data[k] == 2 and preds[k] == 2:
                        class2 += 1

            if count_batch % 10 == 0 and phase == 'train':
                batch_loss = float(running_loss) / counti
                batch_acc = float(running_corrects) / counti
                print('{} Epoch [{}] Batch [{}] Loss: {:.4f} Acc: {:.4f} Time: {:.4f}s'. \
                      format(phase, epoch + 1, count_batch, batch_loss, batch_acc, time.time() - begin_time))
                begin_time = time.time()

        epoch_loss = float(running_loss) / dataset_sizes[phase]
        epoch_acc = float(running_corrects) / dataset_sizes[phase]
        if phase == 'test':
            print(running_corrects)
            print(class0, class1, class2)
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        if phase == 'train':
            if (epoch + 1) % 40 == 0:
                if not os.path.exists('/home/xf/model'):
                    os.makedirs('/home/xf/model')
                torch.save(model, '/home/xf/model/resnet50-lbp944_epoch{}_sal.pkl'.format(epoch + 1))
                torch.save(model2, '/home/xf/model/res-dnn-lbp944_epoch{}_sal.pkl'.format(epoch + 1))
        # deep copy the model
        if phase == 'test':
            if epoch_acc > best_acc:
                best0 = class0
                best1 = class1
                best2 = class2
                best_acc = epoch_acc
                torch.save(model, "/home/xf/bestmodel/resnet50-g-lbp944-sal.pkl")
                torch.save(model2, "/home/xf/bestmodel/res-dnn-lbp944-sal.pkl")

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('Best test Acc: {:4f}'.format(best_acc))
print("best:{},{},{}".format(best0, best1, best2))
