import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from .supervised_learning import SupervisedLearning

class RotNet(SupervisedLearning):
    """RotNet framework"""
    def __init__(self, encoder):
        super().__init__(encoder)
        
        # SSL용 feature 추출을 위해 classifier 제거
        if hasattr(self.encoder, 'classifier'):
            feature_dim = self.encoder.classifier.in_features
            self.encoder.classifier = nn.Identity()
        elif hasattr(self.encoder, 'fc'):
            feature_dim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        else:
            # Get feature dimension
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 32, 32)
                dummy_features = self.encoder._extract_features(dummy_input)
                feature_dim = dummy_features.size(1)
        
        # 4-class rotation classifier
        self.rotation_classifier = nn.Linear(feature_dim, 4)
        
    def forward(self, batch):
        """RotNet forward pass (논문 방식)"""
        if isinstance(batch, tuple):
            x, _ = batch  # Ignore original labels for SSL
        else:
            x = batch
            
        # Create rotation batch (논문: 4개 회전 동시 처리)
        x_rot, y_rot = self._create_rotation_batch(x)
        x_rot = x_rot.to(x.device)
        y_rot = y_rot.to(x.device)
        
        # Extract features and predict rotation
        features = self.encoder._extract_features(x_rot)
        rotation_pred = self.rotation_classifier(features)
        
        # Compute rotation loss
        loss = F.cross_entropy(rotation_pred, y_rot)
        return loss
        
    def extract_features(self, x):
        """Extract features for k-NN evaluation"""
        with torch.no_grad():
            return self.encoder._extract_features(x)
    
    def move_batch_to_device(self, batch, device):
        """Move batch to device"""
        if isinstance(batch, tuple):
            return tuple(item.to(device) if torch.is_tensor(item) else item for item in batch)
        else:
            return batch.to(device)
    
    def _create_rotation_batch(self, x):
        """Create 4 rotation batch (논문 방식)"""
        batch_size = x.size(0)
        
        # 4개 회전 생성
        x_rot0 = x  # 0°
        x_rot1 = torch.rot90(x, 1, dims=[2, 3])  # 90°
        x_rot2 = torch.rot90(x, 2, dims=[2, 3])  # 180°
        x_rot3 = torch.rot90(x, 3, dims=[2, 3])  # 270°
        
        # Stack all rotations
        x_all = torch.cat([x_rot0, x_rot1, x_rot2, x_rot3], dim=0)
        
        # Create rotation labels
        y_rot = torch.cat([
            torch.zeros(batch_size, dtype=torch.long),
            torch.ones(batch_size, dtype=torch.long),
            torch.full((batch_size,), 2, dtype=torch.long),
            torch.full((batch_size,), 3, dtype=torch.long)
        ], dim=0)
        
        return x_all, y_rot
    
    def collect_features(self, data_loader, device):
        """Collect features for k-NN evaluation"""
        self.eval()
        features_list = []
        labels_list = []
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, tuple):
                    x, y = batch
                    x = x.to(device)
                    y = y.to(device)
                    
                    features = self.extract_features(x)
                    features_list.append(features.cpu().numpy())
                    labels_list.append(y.cpu().numpy())
                else:
                    # SSL dataset without labels - skip
                    continue
        
        if not features_list:
            return None, None
            
        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        
        return features, labels
    
    def knn_evaluation(self, train_features, train_labels, test_features, test_labels, k=5):
        """k-NN evaluation (논문: L2/cosine distance)"""
        print(f"Running k-NN with k={k}...")
        
        # L2 distance (Euclidean)
        knn_l2 = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=-1)
        knn_l2.fit(train_features, train_labels)
        pred_l2 = knn_l2.predict(test_features)
        acc_l2 = accuracy_score(test_labels, pred_l2)
        
        # Cosine distance
        knn_cosine = KNeighborsClassifier(n_neighbors=k, metric='cosine', n_jobs=-1)
        knn_cosine.fit(train_features, train_labels)
        pred_cosine = knn_cosine.predict(test_features)
        acc_cosine = accuracy_score(test_labels, pred_cosine)
        
        print(f"k-NN (k={k}, L2): {acc_l2*100:.2f}%")
        print(f"k-NN (k={k}, Cosine): {acc_cosine*100:.2f}%")
        
        return max(acc_l2, acc_cosine)  # Return best result