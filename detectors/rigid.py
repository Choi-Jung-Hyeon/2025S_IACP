from typing import Dict, List, Tuple, Union
import math, os, glob
import torch
from PIL import Image
from .backbones.dino import Dino

def _load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img

def _add_noise(img: Image.Image, noise_std: float) -> Image.Image:
    # 이미지 픽셀공간에서 가우시안 노이즈 추가 (논문 설정)
    import numpy as np
    x = np.asarray(img).astype("float32") / 255.0
    n = np.random.normal(loc=0.0, scale=noise_std, size=x.shape).astype("float32")
    y = np.clip(x + n, 0.0, 1.0)
    y = (y * 255.0).round().astype("uint8")
    return Image.fromarray(y)

class RigidDetector:
    def __init__(
        self,
        hf_model: str = "facebook/dinov2-base",
        device: str = None,
        noise_std: float = 0.10,     # λ (0.05~0.15 권장)
        n_samples: int = 8,          # 섭동 샘플 수 (4~16 권장)
        epsilon: float = 0.985,      # 코사인 유사도 임계값 (캘리브레이션 권장)
    ):
        self.backbone = DinoHF(hf_model, device)
        self.noise_std = noise_std
        self.n_samples = n_samples
        self.epsilon = epsilon
        self.cos = torch.nn.CosineSimilarity(dim=-1)

    @torch.no_grad()
    def score_image(self, img: Image.Image) -> Tuple[float, bool]:
        """
        반환: (유사도 평균, is_generated)
        - 원본 vs 노이즈 이미지 임베딩 코사인 유사도 평균
        - 평균 유사도 <= epsilon 이면 생성(AI)로 판단
        """
        f0 = self.backbone.embed(img)  # (1, D)
        sims = []
        for _ in range(self.n_samples):
            noisy = _add_noise(img, self.noise_std)
            f1 = self.backbone.embed(noisy)
            sim = self.cos(f0, f1).item()
            sims.append(sim)
        s = float(sum(sims) / len(sims))
        return s, (s <= self.epsilon)

    def calibrate(self, real_dir: str, pattern: str = "**/*.*", max_imgs: int = 200) -> float:
        """
        소량의 '실제' 이미지로 epsilon을 자동 산정(예: 실제 95% 통과).
        """
        import numpy as np
        paths = [p for p in glob.glob(os.path.join(real_dir, pattern), recursive=True)
                 if os.path.splitext(p)[1].lower() in [".jpg",".jpeg",".png",".bmp",".webp"]]
        paths = paths[:max_imgs]
        scores = []
        for p in paths:
            img = _load_image(p)
            s, _ = self.score_image(img)
            scores.append(s)
        # 실제 95%가 통과하도록 하한선 설정
        eps = float(np.percentile(scores, 5))  # 하위 5% 지점
        self.epsilon = eps
        return eps

    def run_dir(self, input_path: str) -> List[Dict]:
        """
        폴더/단일 이미지 입력을 모두 지원. 결과 리스트 반환.
        """
        if os.path.isdir(input_path):
            paths = sorted([p for p in glob.glob(os.path.join(input_path, "**/*.*"), recursive=True)
                            if os.path.splitext(p)[1].lower() in [".jpg",".jpeg",".png",".bmp",".webp"]])
        else:
            paths = [input_path]

        out = []
        for p in paths:
            img = _load_image(p)
            s, is_gen = self.score_image(img)
            out.append({"path": p, "similarity": s, "pred": "AI" if is_gen else "REAL"})
        return out

