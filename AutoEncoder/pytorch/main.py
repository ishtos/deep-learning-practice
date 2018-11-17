import os
import sys
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

from utils import dataloader
from autoencoder import Encoder, Decoder, AutoEncoder


def imshow(images, file_name='train'):
    images = images.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    images = std * images + mean
    images = np.clip(images, 0, 1)
    plt.imshow(images)
    plt.savefig('{}.png'.format(file_name))


def main(args):
    ## load datasets
    train_dataset = dataloader('dogs_cats', 'train')
    test_dataset = dataloader('dogs_cats', 'test')

    ## split train and validation
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = 5000

    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    ## train and validation loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler
    )
    valid_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=validation_sampler
    )

    if args.debug:
        images, _ = next(iter(train_loader))
        grid = torchvision.utils.make_grid(images[:25], nrow=5)
        imshow(grid, 'train')

        images, _ = next(iter(valid_loader))
        grid = torchvision.utils.make_grid(images[:25], nrow=5)
        imshow(grid, 'valid')

    model = AutoEncoder()
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print('cuda is available!')
        model.cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=1e-5)

    log_dir = 'logs'
    if not os.path.isdir('logs'):
        os.mkdir('logs')

    if args.mode == 'train':
        best_val = 5
        loss_list = []
        val_loss_list = []
        for epoch in range(args.n_epochs):
            loss = train(model, criterion, optimizer, train_loader, use_gpu)
            val_loss = valid(model, criterion, valid_loader, use_gpu)

            print('epoch %d, loss: %.4f val_loss: %.4f' %
                (epoch, loss, val_loss))

            if val_loss < best_val:
                print('val_loss improved from %.5f to %.5f!' % (best_val, val_loss))
                best_val = val_loss
                model_file = 'epoch%03d-%.3f.pth' % (epoch, val_loss)
                torch.save(model.state_dict(), os.path.join(log_dir, model_file))

            loss_list.append(loss)
            val_loss_list.append(val_loss)
    
    if args.mode == 'test':
        weight_file = args.path_weight_file

        model = AutoEncoder()
        model.load_state_dict(
            torch.load(
                weight_file,
                map_location=lambda storage,
                loc: storage))

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False)

        images, _ = iter(test_loader).next()
        images = Variable(images, volatile=True)
        print(images.size())

        imshow(torchvision.utils.make_grid(images.data[:25], nrow=5))

        outputs = model(images)
        
        imshow(torchvision.utils.make_grid(outputs.data[:25], nrow=5))



def train(model, criterion, optimizer, train_loader, use_gpu):
    model.train()
    running_loss = 0
    for _, (images, _) in enumerate(train_loader):
        if use_gpu:
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, images)
        running_loss += loss.data

        loss.backward()
        optimizer.step()

    train_loss = running_loss / len(train_loader)

    return train_loss


def valid(model, criterion, valid_loader, use_gpu):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    for _, (images, _) in enumerate(valid_loader):
        if use_gpu:
            images = Variable(images.cuda(), volatile=True)
        else:
            images = Variable(images, volatile=True)

        outputs = model(images)

        loss = criterion(outputs, images)
        running_loss += loss.data

    val_loss = running_loss / len(valid_loader)

    return val_loss


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['train', 'test'], help='Mode: train or test', default='train')
    parser.add_argument('--n_epochs', type=int, help='Number of epochs', default=500)
    parser.add_argument('--batch_size', type=int, help='Number of batch size', default=8)
    parser.add_argument('--path_weight_file', type=str, help='Path to weight file')
    parser.add_argument('--debug', type=bool, help='Debug')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))