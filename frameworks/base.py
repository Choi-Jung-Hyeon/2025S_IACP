import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class BaseFramework(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self._remove_classifier()
    
    def _remove_classifier(self):
        pass
    
    def forward(self, batch):
        raise NotImplementedError
    
    def extract_features(self, x):
        with torch.no_grad():
            if hasattr(self.encoder, '_extract_features'):
                return self.encoder._extract_features(x)
            return self.encoder(x)
    
    def move_batch_to_device(self, batch, device):
        if isinstance(batch, tuple):
            return tuple(b.to(device) if torch.is_tensor(b) else b for b in batch)
        return batch.to(device)
    
    def collect_features(self, data_loader, device):
        self.eval()
        features, labels = [], []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = self.move_batch_to_device(batch, device)
                x, y = batch if isinstance(batch, tuple) else (batch, None)
                
                feat = self.extract_features(x)
                features.append(feat.cpu().numpy())
                if y is not None:
                    labels.append(y.cpu().numpy())
        
        features = np.concatenate(features) if features else None
        labels = np.concatenate(labels) if labels else None
        return features, labels

    # 1-nn evaluation
    def knn_evaluation(self, train_features, train_labels, test_features, test_labels):
        train_features = train_features / (np.linalg.norm(train_features, axis=1, keepdims=True) + 1e-8)
        test_features = test_features / (np.linalg.norm(test_features, axis=1, keepdims=True) + 1e-8)
        dists = np.matmul(test_features, train_features.T)
        preds = train_labels[np.argmax(dists, axis=1)]
        accuracy = np.mean(preds == test_labels)
        return accuracy
