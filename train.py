import os
import glob
import numpy as np
from tqdm import tqdm
import cv2
import torch
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
import tensorboardX as tbx
from unet.unet_model import UNet
from mhp import MHP

def train_epoch(train_loader, model, criterion, optimizer, epoch, writer):

    loss_sum = 0
    for img_batch, target_batch in tqdm(train_loader):

        # img = img_batch[0]
        # img = img.numpy().transpose(1, 2, 0)
        # cv2.imshow('', img)
        # cv2.waitKey()

        img_batch = img_batch.cuda()
        target_batch = target_batch.cuda()

        output = model(img_batch)
        loss = criterion(output, target_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

    writer.add_scalar("train loss", loss_sum / len(train_loader), epoch)
    print(loss.item())

    label_pred = torch.max(output[0], axis=0).indices
    label_pred = label_pred.cpu().detach().numpy().astype('uint8')
    cv2.imwrite('img/{0:05d}.jpg'.format(epoch), label_pred)

    target = target_batch[0].cpu().numpy()
    target = target.astype('uint8')
    cv2.imwrite('img/target.jpg', target)

def main():

    train_dataset = MHP('/root/dataset/LV-MHP-v2/train', n_classes=59)
    train_loader = DataLoader(dataset=train_dataset, batch_size=12, shuffle=True, num_workers=0)
    model = UNet(n_channels=3, n_classes=59).cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    writer = tbx.SummaryWriter(log_dir = "logs")
    
    n_epochs = 10000
    for epoch in range(n_epochs):

        train_epoch(train_loader, model, criterion, optimizer, epoch, writer)

        state = {'state_dict': model.state_dict()}
        filename = 'checkpoints/{0:05d}.pth.tar'.format(epoch)
        torch.save(state, filename)

if __name__ == '__main__':
    main()