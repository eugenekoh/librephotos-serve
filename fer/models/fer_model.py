
from fer.models.masking import masking
import torch
import torch.nn as nn
import torch.nn.functional as F

def accuracy(outputs, labels):
    # torch.max outputs max_value, max_indices => here we only care about indices
    _, preds = torch.max(outputs, dim=1)
    # .item() gets a number from a tensor containing a single value
    return torch.tensor(torch.sum(preds == labels).item()/len(preds))

class FERBase(nn.Module):

    # this takes is batch from training dl
    def training_step(self, batch):
        images, labels, fv = batch
        # calls the training model and generates predictions
        out = self(images, fv)
        # calculates loss compare to real labels using cross entropy
        loss = F.cross_entropy(out, labels)
        #acc = accuracy(out, labels)
        return loss  # , acc

    # this takes in batch from validation dl
    def validation_step(self, batch):
        images, labels, fv = batch
        out = self(images, fv)
        loss = F.cross_entropy(out, labels)
        # calls the accuracy function to measure the accuracy
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        # finds out the mean loss of the epoch batch
        epoch_loss = torch.stack(batch_losses).mean()

        batch_accs = [x['val_acc'] for x in outputs]
        # finds out the mean acc of the epoch batch
        epoch_acc = torch.stack(batch_accs).mean()

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result, time):
        print("Epoch [{}], time: {:.2f}, last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, time, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))


def conv_block(in_chnl, out_chnl, pool=False, padding=1):
    layers = [
        nn.Conv2d(in_chnl, out_chnl, kernel_size=3, padding=padding),
        nn.BatchNorm2d(out_chnl),
        nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class FERModel(FERBase):
    def __init__(self, in_chnls, num_cls):
        super().__init__()

        self.conv1 = conv_block(in_chnls, 64, pool=True)           # 64x24x24
        self.conv2 = conv_block(64, 128, pool=True)                # 128x12x12
        self.resnet1 = nn.Sequential(conv_block(128, 128), conv_block(
            128, 128))    # Resnet layer 1: includes 2 conv2d

        self.conv3 = conv_block(128, 256, pool=True)       # 256x6x6
        self.conv4 = conv_block(256, 512, pool=True)       # 512x3x3
        self.resnet2 = nn.Sequential(conv_block(512, 512), conv_block(
            512, 512))    # Resnet layer 2: includes 2 conv2d

        self.mask1 = masking(128, 128, depth=2)
        #self.mask2 = masking(128, 128, depth=2)
        #self.mask3 = masking(256, 256, depth=1)

        self.dropout = nn.Dropout(0.2)
        self.flatten = nn.Sequential(nn.MaxPool2d(3),
                                     nn.Flatten())
        self.fc = nn.Linear(1808, 512)
        self.classifier = nn.Linear(1808, num_cls)
        self.classifier_512 = nn.Linear(512, num_cls)
#         self.classifier = nn.Sequential(nn.MaxPool2d(3),
#                                         nn.Flatten(),
#                                         nn.Linear(512, num_cls))    # num_cls

    def forward(self, xb, fv):

        #print(xb.type())
        #print("xb: ", xb.size())

        #print(fv.size())

        out = self.conv1(xb)
        #print(out.size())

        out = self.conv2(out)
        #print(out.size())
        out = self.resnet1(out) + out
        #print(out.size())

        m = self.mask1(out)
        #print("m is ", m.size())
        out = out * (1+m)
        #print(out.size())

        out = self.conv3(out)
        #print(out.size())
        out = self.conv4(out)
        #print(out.size())
        out = self.resnet2(out) + out
        #print(out.size())
        out = self.flatten(out)
        #print("penul out: ", out.type())

        out = torch.cat((out, fv), 1)
        #print("final out: ", out.size())

        out = self.fc(out)
        #out = self.dropout(out)
        return self.classifier_512(out)