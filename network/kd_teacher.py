import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherModel(nn.Module):
    """Frozen teacher model for knowledge distillation.

    Extracts spatial patch features from V-JEPA2 or DINOv3 ViT and reshapes
    them into (B, hidden_size, H/patch, W/patch) feature maps.

    The teacher is always frozen — no gradients flow through it.
    """

    TEACHER_CONFIGS = {
        'vjepa2': {
            'class': 'VJEPA2Model',
            'default_model': 'facebook/vjepa2-vitl-fpc64-256',
            'hidden_size': 1024,
            'patch_size': 16,
        },
        'dinov3': {
            'class': 'DINOv3ViTModel',
            'default_model': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
            'hidden_size': 768,
            'patch_size': 16,
        },
    }

    def __init__(self, teacher_name: str, model_id: str | None = None,
                 teacher_img_size: int = 224):
        super().__init__()
        self.teacher_name = teacher_name
        info = self.TEACHER_CONFIGS[teacher_name]
        model_id = model_id or info['default_model']
        self.hidden_size = info['hidden_size']
        self.patch_size = info['patch_size']
        self.teacher_img_size = teacher_img_size

        self.model = self._load_model(teacher_name, model_id, info)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @staticmethod
    def _load_model(teacher_name, model_id, info):
        if teacher_name == 'vjepa2':
            from transformers import VJEPA2Model
            return VJEPA2Model.from_pretrained(model_id)
        elif teacher_name == 'dinov3':
            from transformers import DINOv3ViTModel
            return DINOv3ViTModel.from_pretrained(model_id)
        raise ValueError(f"Unknown teacher: {teacher_name}")

    @staticmethod
    def from_config(model_id: str):
        """Auto-detect teacher type from HuggingFace model config."""
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id)
        if cfg.model_type == 'vjepa2':
            return 'vjepa2'
        if cfg.model_type in ('dinov3_vit',):
            return 'dinov3'
        raise ValueError(f"Cannot auto-detect teacher type for {model_id} "
                         f"(model_type={cfg.model_type})")

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Prepare student input for the teacher.

        Handles: grayscale→RGB, resize to teacher_img_size.
        """
        B, C, H, W = x.shape
        if C == 1:
            x = x.expand(-1, 3, -1, -1)
        ts = self.teacher_img_size
        if H != ts or W != ts:
            x = F.interpolate(x, size=(ts, ts), mode='bilinear', align_corners=False)
        return x

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature map from teacher.

        Args:
            x: student input tensor (B, C, H, W), can be 1-ch or 3-ch.

        Returns:
            Spatial features (B, hidden_size, h, w) where h=w=teacher_img_size/patch_size.
        """
        x = self._prepare_input(x)
        h = w = self.teacher_img_size // self.patch_size

        if self.teacher_name == 'vjepa2':
            x_vid = x.unsqueeze(1)  # (B, 1, 3, H, W)
            out = self.model(x_vid, skip_predictor=True)
            patches = out.last_hidden_state  # (B, num_patches, D)
        elif self.teacher_name == 'dinov3':
            out = self.model(pixel_values=x)
            tokens = out.last_hidden_state  # (B, 1+reg+patches, D)
            n_reg = getattr(self.model.config, 'num_register_tokens', 0)
            patches = tokens[:, 1 + n_reg:, :]  # skip CLS + register
        else:
            raise ValueError(f"Unknown teacher: {self.teacher_name}")

        spatial = patches.reshape(patches.shape[0], h, w, self.hidden_size)
        spatial = spatial.permute(0, 3, 1, 2).contiguous()  # (B, D, h, w)
        return spatial
