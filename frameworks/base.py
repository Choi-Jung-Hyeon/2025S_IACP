import torch
import torch.nn as nn
import torch.nn.functional as F

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
        if hasattr(self.encoder, '_extract_features'):
            return self.encoder._extract_features(x)
        return self.encoder(x)

    def move_batch_to_device(self, batch, device):
        if isinstance(batch, (tuple, list)):
            return type(batch)(b.to(device) if torch.is_tensor(b) else b for b in batch)
        if torch.is_tensor(batch):
            return batch.to(device)
        return batch

    def collect_features(self, data_loader, device):
        self.eval()
        features, labels = [], []

        with torch.no_grad():
            for batch in data_loader:
                batch = self.move_batch_to_device(batch, device)
                
                if isinstance(batch, (tuple, list)):
                    x, y = batch
                else:
                    x, y = batch, None
                
                feat = self.extract_features(x)
                features.append(feat)
                if y is not None:
                    labels.append(y)
        
        features = torch.cat(features, dim=0) if features else None
        labels = torch.cat(labels, dim=0) if labels else None
        return features, labels

    def knn_evaluation(self, train_features, train_labels, test_features, test_labels):
        train_features = F.normalize(train_features, p=2, dim=1)
        test_features = F.normalize(test_features, p=2, dim=1)
        
        dists = torch.matmul(test_features, train_features.T)
        
        preds_indices = torch.argmax(dists, dim=1)
        preds = train_labels[preds_indices]
        
        accuracy = (preds == test_labels).float().mean()
        
        return accuracy
