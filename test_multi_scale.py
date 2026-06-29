#!/usr/bin/env python3
"""
Quick test script to verify multi-scale Restormer architecture and loss.
This tests:
1. PhaseFormerUpsampler module
2. Restormer with multi-scale output enabled
3. PhaseFormerMultiScaleLoss computation
"""

import torch
import sys

# Test 1: Import and verify modules exist
print("=" * 60)
print("Test 1: Importing modules...")
print("=" * 60)

try:
    from basicsr.models.archs.restormer_arch import Restormer, PhaseFormerUpsampler
    print("✓ Successfully imported Restormer and PhaseFormerUpsampler")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

try:
    from basicsr.models.losses.losses import PhaseFormerMultiScaleLoss
    print("✓ Successfully imported PhaseFormerMultiScaleLoss")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Test 2: Test PhaseFormerUpsampler module
print("\n" + "=" * 60)
print("Test 2: Testing PhaseFormerUpsampler module...")
print("=" * 60)

try:
    upsampler = PhaseFormerUpsampler(96)
    x = torch.randn(2, 96, 128, 128)  # (B, 96, H, W)
    y = upsampler(x)
    assert y.shape == (2, 48, 256, 256), f"Expected (2, 48, 256, 256), got {y.shape}"
    print(f"✓ PhaseFormerUpsampler: {x.shape} → {y.shape}")
except Exception as e:
    print(f"✗ PhaseFormerUpsampler test failed: {e}")
    sys.exit(1)

# Test 3: Test Restormer with single-scale output (default)
print("\n" + "=" * 60)
print("Test 3: Testing Restormer with single-scale output...")
print("=" * 60)

try:
    model_single = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        enable_multi_scale_output=False,
        dual_pixel_task=False
    )
    model_single.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y = model_single(x)
    assert isinstance(y, torch.Tensor), "Expected tensor output for single-scale mode"
    assert y.shape == (1, 3, 256, 256), f"Expected (1, 3, 256, 256), got {y.shape}"
    print(f"✓ Single-scale mode: {x.shape} → {y.shape}")
except Exception as e:
    print(f"✗ Single-scale test failed: {e}")
    sys.exit(1)

# Test 4: Test Restormer with multi-scale output
print("\n" + "=" * 60)
print("Test 4: Testing Restormer with multi-scale output...")
print("=" * 60)

try:
    model_multi = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        enable_multi_scale_output=True,
        dual_pixel_task=False
    )
    model_multi.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        outputs = model_multi(x)
    
    assert isinstance(outputs, list), "Expected list output for multi-scale mode"
    assert len(outputs) == 2, f"Expected 2 outputs, got {len(outputs)}"
    
    img_1x, img_2x = outputs
    assert img_1x.shape == (1, 3, 256, 256), f"Expected 1x: (1, 3, 256, 256), got {img_1x.shape}"
    assert img_2x.shape == (1, 3, 512, 512), f"Expected 2x: (1, 3, 512, 512), got {img_2x.shape}"
    print(f"✓ Multi-scale mode:")
    print(f"  1x output: {x.shape} → {img_1x.shape}")
    print(f"  2x output: {x.shape} → {img_2x.shape}")
except Exception as e:
    print(f"✗ Multi-scale test failed: {e}")
    sys.exit(1)

# Test 5: Test PhaseFormerMultiScaleLoss
print("\n" + "=" * 60)
print("Test 5: Testing PhaseFormerMultiScaleLoss...")
print("=" * 60)

try:
    loss_fn = PhaseFormerMultiScaleLoss(
        loss_weight=1.0,
        reduction='mean',
        weight_1x=0.5,
        weight_2x=0.5
    )
    
    pred_1x = torch.randn(2, 3, 256, 256, requires_grad=True)
    pred_2x = torch.randn(2, 3, 512, 512, requires_grad=True)
    target = torch.randn(2, 3, 256, 256)
    
    loss = loss_fn([pred_1x, pred_2x], target)
    assert isinstance(loss, torch.Tensor), "Expected tensor loss output"
    assert loss.requires_grad, "Loss should require gradients"
    
    # Test backward pass
    loss.backward()
    assert pred_1x.grad is not None, "1x prediction should have gradients"
    assert pred_2x.grad is not None, "2x prediction should have gradients"
    
    print(f"✓ Loss computation: {loss.item():.6f}")
    print(f"✓ Backward pass successful")
except Exception as e:
    print(f"✗ Loss test failed: {e}")
    sys.exit(1)

# Test 6: Test with dual_pixel_task (should not enable multi-scale)
print("\n" + "=" * 60)
print("Test 6: Testing dual_pixel_task compatibility...")
print("=" * 60)

try:
    model_dual = Restormer(
        inp_channels=6,
        out_channels=3,
        dim=48,
        enable_multi_scale_output=True,  # This should be ignored
        dual_pixel_task=True
    )
    model_dual.eval()
    x = torch.randn(1, 6, 256, 256)
    with torch.no_grad():
        y = model_dual(x)
    
    # Should return single tensor, not list (dual_pixel_task takes precedence)
    assert isinstance(y, torch.Tensor), "Dual-pixel task should return single tensor"
    assert y.shape == (1, 3, 256, 256), f"Expected (1, 3, 256, 256), got {y.shape}"
    print(f"✓ Dual-pixel task (inp: {x.shape}) → {y.shape}")
    print(f"  Note: enable_multi_scale_output is ignored when dual_pixel_task=True")
except Exception as e:
    print(f"✗ Dual-pixel test failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
print("\nSummary:")
print("  ✓ PhaseFormerUpsampler working correctly")
print("  ✓ Single-scale output mode working")
print("  ✓ Multi-scale output mode working (1x + 2x)")
print("  ✓ PhaseFormerMultiScaleLoss working with gradients")
print("  ✓ Dual-pixel task compatibility maintained")
