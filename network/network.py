import torch
import torch.nn as nn

from .encoder.pvtv2 import pvt_v2_b2, pvt_v2_b0, pvt_v2_b1, pvt_v2_b3
from .edge_net import EdgeNet, SobelEdgeDetector, CannyEdgeDetector, ConcatFusion
from .edge_attention import EdgeCrossAttention
from .decoder import Decoder

BACKBONE_REGISTRY = {
    'pvt_v2_b0': (pvt_v2_b0, [32, 64, 160, 256]),
    'pvt_v2_b1': (pvt_v2_b1, [64, 128, 320, 512]),
    'pvt_v2_b2': (pvt_v2_b2, [64, 128, 320, 512]),
    'pvt_v2_b3': (pvt_v2_b3, [64, 128, 320, 512]),
}

EDGE_BUILDERS = {
    'learned': lambda ch: EdgeNet(in_ch=3, mid_ch=64, enc3_ch=ch[2], enc4_ch=ch[3]),
    'sobel':   lambda ch: SobelEdgeDetector(enc3_ch=ch[2], enc4_ch=ch[3]),
    'canny':   lambda ch: CannyEdgeDetector(enc3_ch=ch[2], enc4_ch=ch[3]),
}


class EdgeGuidedSegModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, backbone='pvt_v2_b2',
                 num_heads=8, attn_drop=0., proj_drop=0.,
                 edge_type='learned', attention_type='cross_attention',
                 q_source='encoder', attention_scales=None,
                 deep_supervision=False):
        super().__init__()
        self.in_channels = in_channels
        self.edge_type = edge_type
        self.attention_type = attention_type
        self.q_source = q_source
        self.attention_scales = attention_scales or ['enc3', 'enc4']
        self.deep_supervision = deep_supervision

        if in_channels != 3:
            self.input_proj = nn.Conv2d(in_channels, 3, kernel_size=1)
        else:
            self.input_proj = None

        backbone_cls, channels = BACKBONE_REGISTRY[backbone]
        self.encoder = backbone_cls()
        self.channels = channels

        # --- Edge branch ---
        if edge_type in EDGE_BUILDERS:
            self.edge_net = EDGE_BUILDERS[edge_type](channels)
        else:
            self.edge_net = None

        # --- Fusion at enc3 scale ---
        if attention_type == 'cross_attention' and 'enc3' in self.attention_scales:
            self.mhsa3 = EdgeCrossAttention(
                dim=channels[2], num_heads=num_heads,
                attn_drop=attn_drop, proj_drop=proj_drop,
            )
        elif attention_type == 'concat' and edge_type != 'none':
            self.mhsa3 = ConcatFusion(channels[2])
        else:
            self.mhsa3 = None

        # --- Fusion at enc4 scale ---
        if attention_type == 'cross_attention' and 'enc4' in self.attention_scales:
            self.mhsa4 = EdgeCrossAttention(
                dim=channels[3], num_heads=num_heads,
                attn_drop=attn_drop, proj_drop=proj_drop,
            )
        elif attention_type == 'concat' and edge_type != 'none':
            self.mhsa4 = ConcatFusion(channels[3])
        else:
            self.mhsa4 = None

        self.decoder = Decoder(channels, num_classes, deep_supervision=deep_supervision)

    def forward(self, x, return_features=False):
        if self.input_proj is not None:
            x = self.input_proj(x)

        f1, f2, f3, f4 = self.encoder(x)

        a3, a4 = f3, f4

        if self.edge_net is not None:
            e3, e4 = self.edge_net(x)

            if self.mhsa3 is not None:
                if self.attention_type == 'cross_attention':
                    if self.q_source == 'encoder':
                        a3 = self.mhsa3(f3, e3)
                    else:
                        a3 = self.mhsa3(e3, f3)
                else:
                    a3 = self.mhsa3(f3, e3)

            if self.mhsa4 is not None:
                if self.attention_type == 'cross_attention':
                    if self.q_source == 'encoder':
                        a4 = self.mhsa4(f4, e4)
                    else:
                        a4 = self.mhsa4(e4, f4)
                else:
                    a4 = self.mhsa4(f4, e4)

        logits = self.decoder(a4, a3, f2, f1)
        if return_features:
            return logits, {'f3': f3, 'f4': f4}
        return logits

    def load_pretrained_backbone(self, path):
        state = torch.load(path, map_location='cpu')
        self.encoder.load_state_dict(state, strict=False)
