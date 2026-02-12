import torch
import pytest
import numpy as np
from PIL import Image
from collections import OrderedDict
from models.explainer import Explainer

def test_explainer_vit_integration():
    """
    Tests that Explainer works correctly with Vision Transformer (ViT) architecture.
    """
    
    # Mock ViT with Sequence Output and Dict Return
    class MockViT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vit = torch.nn.Module()
            self.vit.layernorm = torch.nn.LayerNorm(768)
            
            # Mock Encoder Layer Structure
            self.vit.encoder = torch.nn.Module()
            self.vit.encoder.layer = torch.nn.ModuleList([
                torch.nn.Module() for _ in range(12)
            ])
            # Add layernorm_before to the last layer
            self.vit.encoder.layer[-1].layernorm_before = torch.nn.LayerNorm(768)
            
            self.classifier = torch.nn.Linear(768, 2)
            
        def forward(self, x):
            # Simulate ViT output: (Batch, 197, 768)
            B = x.shape[0]
            seq_output = torch.randn(B, 197, 768, requires_grad=True)
            
            # Pass through the NEW target layer
            # This ensures the hook captures the tensor
            feat = self.vit.encoder.layer[-1].layernorm_before(seq_output)
            
            # Continue flow (dummy)
            norm_feat = self.vit.layernorm(feat)
            cls_token = norm_feat[:, 0]
            logits = self.classifier(cls_token)
            
            class MockOutput(OrderedDict):
                def __init__(self, logits):
                    super().__init__()
                    self.logits = logits
                    self['logits'] = logits
            return MockOutput(logits)

    model = MockViT()
    explainer = Explainer(model)
    
    input_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)
    original_image = Image.new('RGB', (224, 224))
    
    heatmap = explainer.generate_heatmap(input_tensor, 0, original_image)
    
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (224, 224, 3)
