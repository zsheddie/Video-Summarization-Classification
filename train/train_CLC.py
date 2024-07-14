import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.CLC import CNN_LSTM
from sklearn.cluster import KMeans


def train_step(model, data_loader, num_epochs, optimizer, device = None):
    
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        #for i, (videos, label) in enumerate(data_loader):
        for videos, label in tqdm(data_loader):

            videos = videos.to(device, non_blocking=True)
            label = label.long().to(device, non_blocking=True)
            #print(label.shape)
            out = model(videos).to(device)
            loss = criterion(out, label) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            #if i % 10 == 0:
            #   print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item()}')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(data_loader):.4f}')



def test_step(model, data_loader, num_epochs, optimizer, device = None):
    criterion = nn.CrossEntropyLoss()
    running_loss = 0
    model.eval()
    pred = []
    for i, (videos, label) in enumerate(data_loader):
        videos = videos.to(device)
        label = label.long().to(device)
        out = model(videos)
        loss = criterion(out, label) 
        
        pred.append(out.detach().cpu())
        running_loss += loss.item()
        #if i % 10 == 0:
    print(f'Loss: {running_loss/len(data_loader):.4f}')

    pred = torch.cat(pred, dim=0).numpy()
    return pred