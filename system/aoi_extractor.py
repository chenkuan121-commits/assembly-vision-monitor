"""
AOI 标准件特征比对模块 (Golden Sample Matching)
使用 torchvision 预训练 ResNet18 提取归一化特征向量，余弦相似度比对。
支持多角度比对，容忍工件放置时的轻微旋转偏差。
"""
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
import cv2

try:
    from torchvision.models import ResNet18_Weights, MobileNet_V3_Small_Weights
except ImportError:
    ResNet18_Weights = None
    MobileNet_V3_Small_Weights = None


class AOIFeatureExtractor:
    SUPPORTED_BACKBONES = ('resnet18', 'mobilenet_v3_small')
    # 多角度比对的旋转角度（度），0° 为原始朝向
    CHECK_ANGLES = (0, -5, 5, -10, 10)

    def __init__(self, backbone='resnet18', device='cpu'):
        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(f"Backbone must be one of {self.SUPPORTED_BACKBONES}")
        self.backbone = backbone
        self.device = device

        if backbone == 'resnet18':
            if ResNet18_Weights is not None:
                model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            else:
                model = models.resnet18(pretrained=True)
            self.feature_dim = 512
        else:
            if MobileNet_V3_Small_Weights is not None:
                model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            else:
                model = models.mobilenet_v3_small(pretrained=True)
            self.feature_dim = 576

        if backbone == 'resnet18':
            self.model = torch.nn.Sequential(*list(model.children())[:-1])
        else:
            self.model = torch.nn.Sequential(model.features, model.avgpool)

        self.model = self.model.to(device)
        self.model.eval()

        # 基础变换（不含旋转，旋转在 extract 时按需叠加）
        self.base_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # ── 内部：单次前向 ──
    def _forward(self, pil_image):
        with torch.no_grad():
            tensor = self.base_transform(pil_image).unsqueeze(0).to(self.device)
            features = self.model(tensor)
            features = features.flatten(1)
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            return features.squeeze(0).cpu().numpy().astype(np.float32)

    # ── 裁剪辅助 ──
    @staticmethod
    def _crop(image_bgr, bbox, pad_ratio=0.05):
        """从 BGR 图中按 bbox 裁剪，带微量外扩"""
        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        bw, bh = x2 - x1, y2 - y1
        pad_w, pad_h = int(bw * pad_ratio), int(bh * pad_ratio)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        if x2 <= x1 or y2 <= y1:
            return None
        return image_bgr[y1:y2, x1:x2]

    # ── 单次提取（建档时用） ──
    def extract(self, image_bgr, bbox=None):
        """从 BGR 图像中提取 L2 归一化特征向量（0° 单次）"""
        if bbox is not None:
            crop = self._crop(image_bgr, bbox)
            if crop is None:
                return np.zeros(self.feature_dim, dtype=np.float32)
        else:
            crop = image_bgr
        pil_img = T.ToPILImage()(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        return self._forward(pil_img)

    # ── 多角度比对（检测时用，容忍工件旋转） ──
    def compare_multi_angle(self, image_bgr, bbox, standard_vector):
        """
        对裁剪区域做多角度旋转后提取特征，返回 (max_similarity, best_angle)。
        角度集合: CHECK_ANGLES = (0°, -5°, +5°, -10°, +10°)
        """
        crop = self._crop(image_bgr, bbox)
        if crop is None:
            return 0.0, 0

        pil_img = T.ToPILImage()(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        best_sim = -1.0
        best_angle = 0

        for angle in self.CHECK_ANGLES:
            if angle == 0:
                rotated = pil_img
            else:
                rotated = T.functional.rotate(pil_img, angle, expand=False)
            vec = self._forward(rotated)
            sim = float(np.dot(vec, standard_vector))
            if sim > best_sim:
                best_sim = sim
                best_angle = angle

        return best_sim, best_angle

    @staticmethod
    def cosine_similarity(vec1, vec2):
        """两个已 L2 归一化向量的余弦相似度 (等价于点积)"""
        return float(np.dot(vec1, vec2))
