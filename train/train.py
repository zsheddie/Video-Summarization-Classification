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
from models.dec import CNN_LSTM, DEC
from sklearn.cluster import KMeans


def train_step(model, dec_model, data_loader, num_epochs=20, n_clusters=20):
    optimizer = optim.Adam(list(model.parameters()) + list(dec_model.parameters()), lr=1e-3)
    criterion = nn.MSELoss()
    
    features = []
    for inputs in data_loader:
        #print(inputs.shape)
        inputs = inputs.cuda()
        #with torch.no_grad():
        #    features.append(model(inputs).cpu().numpy())
        features.append(model(inputs).detach().cpu().numpy())
        
    features = np.concatenate(features, axis=0)

    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(features)
    dec_model.cluster_centers.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).cuda()

    for epoch in range(num_epochs):
        for inputs in tqdm(data_loader):
            #print(inputs.shape)
            #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            inputs = inputs.cuda()
            optimizer.zero_grad()
            features = model(inputs)
            q = dec_model(features)
            p = dec_model.target_distribution(q).detach()
            #print(q.shape, p.shape)
            kl_loss = dec_model.dec_loss(q, p)
            kl_loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], KL Loss: {kl_loss.item():.4f}')

    return q, p

def test_step(model, dec_model, data_loader):
    all_q = []
    all_p = []
    for inputs in tqdm(data_loader):
        #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        inputs = inputs.cuda()
        features = model(inputs)
        q = dec_model(features)
        p = dec_model.target_distribution(q).detach()
        all_q.extend(q.cpu().numpy())
        all_p.extend(p.cpu().numpy())
        #print(q.shape, p.shape)
    
    return all_q, all_p