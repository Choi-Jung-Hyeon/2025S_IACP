from .supervised_learning import SupervisedLearning
import torch.nn as nn
import torch
import torch.nn.functional as F

class SimCLR(SupervisedLearning):
    def __init__(self, encoder, projection_dim=128, temperature=0.5):
        super().__init__(encoder)
        self.encoder = encoder
        feature_dim = encoder.fc.in_features
        encoder.fc = nn.Identity()
        
        # Projection head (MLP)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim)
        )
        self.temperature = temperature
        
    def forward(self, batch):
        (x_i, x_j), _ = batch  # 두 augmented 뷰
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)
        
        z_i = self.projector(h_i)
        z_j = self.projector(h_j)

        loss = self.nt_xent_loss(z_i, z_j)
        return loss

    def move_batch_to_device(self, batch, device):
        (x_i, x_j), y = batch
        x_i = x_i.to(device)
        x_j = x_j.to(device)
        y = y.to(device)
        return (x_i, x_j), y
    
    def nt_xent_loss(self, z_i, z_j):
        batch_size = z_i.shape[0]
        
        z = torch.cat([z_i, z_j], dim=0)

        sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)  
        
        sim_matrix /= self.temperature
        
        labels = (torch.arange(batch_size) + batch_size).to(z.device)
        labels = torch.cat([labels, torch.arange(batch_size).to(z.device)], dim=0)

        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    def collect_features(self, data_loader, device):
        self.eval()
        features, labels = [], []
	
        with torch.no_grad():
            for batch in data_loader:
                x_i, y = batch
                x_i, y = x_i.to(device), y.to(device)
                feat = self.extract_features(x_i)
                features.append(feat)
                if y is not None:
                    labels.append(y)

        features = torch.cat(features, dim=0) if features else None
        labels = torch.cat(labels, dim=0) if labels else None
        return features, labels
