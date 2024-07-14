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
from models.CLK import CNN_LSTM, CLK
from sklearn.cluster import KMeans


def train_step(model, data_loader, num_epochs, optimizer, device = None):
    
    features = []
    for inputs in data_loader:
        #print(inputs.shape)
        inputs = inputs.cuda()
        #with torch.no_grad():
        #    features.append(model(inputs).cpu().numpy())
        features.append(model(inputs).detach().cpu().numpy())
        
    features = np.concatenate(features, axis=0)
    model.initialize_cluster_centers(data_loader)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        for i, videos in enumerate(data_loader):
            videos = videos.to(device)
            features = model(videos)
            #reconstructed_videos = model(videos)
            #loss = criterion(reconstructed_videos, videos) 
            kmeans_loss, cluster_assignments = model.compute_kmeans_loss(features)
            print(cluster_assignments)

            optimizer.zero_grad()
            kmeans_loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], Loss: {kmeans_loss.item()}')


def test_step(model, data_loader, num_epochs, optimizer, device = None):
    
    features = []
    for inputs in data_loader:
        #print(inputs.shape)
        inputs = inputs.cuda()
        #with torch.no_grad():
        #    features.append(model(inputs).cpu().numpy())
        features.append(model(inputs).detach().cpu().numpy())
        
    features = np.concatenate(features, axis=0)
    # model.initialize_cluster_centers(data_loader)

    model.eval()
    pred = []
    for i, videos in enumerate(data_loader):
        videos = videos.to(device)
        features = model(videos)
        kmeans_loss, cluster_assignments = model.compute_kmeans_loss(features)
        print(cluster_assignments)
        pred.append(cluster_assignments.detach().cpu())

        optimizer.zero_grad()
        kmeans_loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], Loss: {kmeans_loss.item()}')
    return pred