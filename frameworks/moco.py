import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .supervised_learning import SupervisedLearning

class MoCo(SupervisedLearning):
    '''
    SimCLR 일정부분 재사용
    q = query = i(SimCLR)   backpropagation으로 업데이트
    k = key = j(SimCLR)     momentum으로 업데이트
    변수명 뒤에 '_k'가 있으면 key, 아무것도 없으면 query
    '''
    def __init__(self, encoder, projection_dim=128, K=4096, momentum=0.999, temperature=0.07):
        super().__init__(encoder)
        self.encoder_q = encoder
        feature_dim = encoder.fc.in_features
        self.encoder_q.fc = nn.Identity()

        self.K = K
        self.m = momentum
        self.temperature = temperature

        # key encoders를 동일 구조로 복제, key는 no-grad
        self.encoder_k = copy.deepcopy(self.encoder_q)
        for parameter in self.encoder_k.parameters():
            parameter.requires_grad = False

        # projection heads (2-layer MLP)
        self.projector_q = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim)
        )

        self.projector_k = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim)
        )
        for parameter in self.projector_k.parameters():
            parameter.requires_grad = False

        # queue: (dim, K)로 보관
        self.register_buffer("queue", torch.randn(projection_dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # BN 일관성: key encoder는 항상 eval BN 사용 (단일 GPU 기준)
        self.encoder_k.eval()

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # keys: (N, C) normalized
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr.item())

        # 큐가 꽉 차면 돌려쓰기 (원형 버퍼)
        if ptr + batch_size <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K
        else:
            first = self.K - ptr
            self.queue[:, ptr:] = keys[:first].T
            self.queue[:, :batch_size - first] = keys[first:].T
            ptr = (batch_size - first) % self.K

        self.queue_ptr[0] = ptr

    def forward(self, batch):
        (x_q, x_k), _ = batch

        h_q = self.encoder_q(x_q)
        h_k = self.encoder_k(x_k)

        z_q = self.projector_q(h_q)
        z_k = self.projector_k(h_k)

        z_q = F.normalize(z_q, dim=1, eps=1e-12)
        z_k = F.normalize(z_k, dim=1, eps=1e-12)

        # 3) 로짓 계산
        # 양성: q·k (N,1), 음성: q·queue (N,K)
        # queue는 (D,K) 이므로 q( N,D ) @ queue( D,K ) = (N,K)
        l_pos = torch.einsum('nd,nd->n', [z_q, z_k]).unsqueeze(-1)  # (N,1)

        with torch.no_grad():
            queue_snapshot = self.queue.clone().detach()

        l_neg = torch.einsum('nd,dk->nk', [z_q, queue_snapshot])      # (N,K)

        logits = torch.cat([l_pos, l_neg], dim=1)               # (N, 1+K)
        logits /= self.temperature

        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)  # 양성 인덱스=0
        loss = F.cross_entropy(logits, labels)
        
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = p_k.data * self.m + p_q.data * (1. - self.m)
        for p_q, p_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            p_k.data = p_k.data * self.m + p_q.data * (1. - self.m)

        # 4) 큐 업데이트
        with torch.no_grad():
            self._dequeue_and_enqueue(z_k)

        return loss

    def move_batch_to_device(self, batch, device):
        (x_q, x_k), y = batch
        x_q = x_q.to(device)
        x_k = x_k.to(device)
        y = y.to(device)
        return (x_q, x_k), y

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

