"""
Tests for model components.
"""

import pytest
import torch


class TestImageEncoder:
    """Tests for ImageEncoder."""
    
    def test_import(self):
        """Test that ImageEncoder can be imported."""
        from src.models import ImageEncoder
        assert ImageEncoder is not None
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_forward_pass(self):
        """Test forward pass of ImageEncoder."""
        # TODO: Implement when torchxrayvision is available
        pass


class TestTextEncoder:
    """Tests for TextEncoder."""
    
    def test_import(self):
        """Test that TextEncoder can be imported."""
        from src.models import TextEncoder
        assert TextEncoder is not None


class TestFusion:
    """Tests for fusion modules."""
    
    def test_concat_fusion(self):
        """Test ConcatFusion."""
        from src.models import ConcatFusion
        
        fusion = ConcatFusion(
            image_dim=1024,
            text_dim=768,
            output_dim=512,
        )
        
        image_features = torch.randn(2, 1024)
        text_features = torch.randn(2, 768)
        
        output = fusion(image_features, text_features)
        assert output.shape == (2, 512)
    
    def test_attention_fusion(self):
        """Test AttentionFusion."""
        from src.models import AttentionFusion
        
        fusion = AttentionFusion(
            image_dim=1024,
            text_dim=768,
            output_dim=512,
        )
        
        image_features = torch.randn(2, 1024)
        text_features = torch.randn(2, 768)
        
        output = fusion(image_features, text_features)
        assert output.shape == (2, 512)


class TestMultimodalClassifier:
    """Tests for MultimodalClassifier."""
    
    def test_import(self):
        """Test that MultimodalClassifier can be imported."""
        from src.models import MultimodalClassifier
        assert MultimodalClassifier is not None
