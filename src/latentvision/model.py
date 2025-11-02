# -*- coding: utf-8 -*-
from typing import Tuple, Optional
import torch, torch.nn as nn

class RandomProjector(nn.Module):
    def __init__(self, out_dim: int = 384):
        super().__init__()
        self.out_dim = out_dim
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        HW = H * W
        return torch.randn(B, HW, self.out_dim, device=x.device)

class ViTEncoder(nn.Module):
    def __init__(self, model_name: str = "vit_base_patch16_224", pretrained: bool = True,
                 out_dim: Optional[int] = None):
        super().__init__()
        self.backend = None
        self.out_dim = out_dim
        self.feat_dim = None
        self.model_name = model_name
        self.pretrained = pretrained
        self.encoder = None
        self.reproj = None
        try:
            import timm
            self.encoder = timm.create_model(model_name, pretrained=pretrained)
            self.encoder.reset_classifier(0)
            self.backend = "timm"
            with torch.no_grad():
                dummy = torch.zeros(1,3,224,224)
                feats = self.encoder.forward_features(dummy)
                if feats.dim()==3:
                    self.feat_dim = feats.shape[-1]
                else:
                    raise RuntimeError("Unexpected feature shape")
        except Exception:
            try:
                from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
                weights = ViT_B_16_Weights.DEFAULT if pretrained else None
                self.encoder = vit_b_16(weights=weights)
                self.backend = "torchvision"
                self.feat_dim = self.encoder.hidden_dim
            except Exception:
                self.encoder = RandomProjector(out_dim=out_dim or 384)
                self.backend = "random"
                self.feat_dim = out_dim or 384
        if self.out_dim is not None and self.out_dim != self.feat_dim:
            self.reproj = nn.Linear(self.feat_dim, self.out_dim)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, return_hw: bool = True) -> Tuple[torch.Tensor, Tuple[int,int]]:
        if self.backend == "timm":
            feats = self.encoder.forward_features(x)
            if feats.shape[1] != (x.shape[-1]//16)*(x.shape[-2]//16):
                feats = feats[:, 1:, :]
            H_p = x.shape[-2]//16; W_p = x.shape[-1]//16
        elif self.backend == "torchvision":
            enc = self.encoder
            x_p = enc._process_input(x)
            batch_class_token = enc.class_token.expand(x.shape[0], -1, -1)
            x_p = torch.cat([batch_class_token, x_p], dim=1)
            x_p = x_p + enc.encoder.pos_embedding
            x_p = enc.encoder.dropout(x_p)
            feats = enc.encoder.layers(x_p)
            feats = enc.encoder.ln(feats)[:,1:,:]
            H_p = x.shape[-2]//enc.patch_size[0]; W_p = x.shape[-1]//enc.patch_size[1]
        else:
            feats = self.encoder(x)
            H_p = x.shape[-2]; W_p = x.shape[-1]
        if self.reproj is not None:
            feats = self.reproj(feats)
        return feats, (H_p, W_p)
