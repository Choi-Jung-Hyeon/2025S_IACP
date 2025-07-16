import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors

class SupervisedLearning(nn.Module):
    """Base class for SSL frameworks - contains common functionality"""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.remove_classifier()
        
    def remove_classifier(self):
        """Remove final classifier for feature extraction"""
        if hasattr(self.encoder, 'fc'):
            self.original_fc = self.encoder.fc
            self.encoder.fc = nn.Identity()
        elif hasattr(self.encoder, 'classifier'):
            self.original_fc = self.encoder.classifier
            self.encoder.classifier = nn.Identity()
            
    def extract_features(self, x):
        """Extract features using encoder backbone"""
        with torch.no_grad():
            features = self.encoder(x)
            return features.view(features.size(0), -1)  # Flatten
            
    def collect_features(self, data_loader, device):
        """Collect features and labels for evaluation"""
        self.eval()
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, tuple):
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                else:
                    x = batch.to(device)
                    y = None
                    
                features = self.extract_features(x)
                all_features.append(features.cpu().numpy())
                if y is not None:
                    all_labels.append(y.cpu().numpy())
                    
        features = np.vstack(all_features)
        labels = np.hstack(all_labels) if all_labels else None
        return features, labels
        
    def knn_evaluation(self, train_features, train_labels, test_features, test_labels, k=5):
        """k-NN classifier evaluation"""
        knn = NearestNeighbors(n_neighbors=k, metric='cosine')
        knn.fit(train_features)
        
        distances, indices = knn.kneighbors(test_features)
        
        correct = 0
        for i, neighbors in enumerate(indices):
            neighbor_labels = train_labels[neighbors]
            predicted_label = np.bincount(neighbor_labels).argmax()
            if predicted_label == test_labels[i]:
                correct += 1
                
        accuracy = correct / len(test_labels)
        return accuracy
        
    def move_batch_to_device(self, batch, device):
        """Move batch to device (handles tuple or tensor)"""
        if isinstance(batch, tuple):
            return tuple(item.to(device) for item in batch)
        return batch.to(device)
        
    def forward(self, batch):
        """Override in child classes"""
        raise NotImplementedError