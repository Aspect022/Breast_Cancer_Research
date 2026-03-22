#!/usr/bin/env python3
"""
Quick Test Script for Fixed Models
Run this to verify all bug fixes work before full training
"""

import torch
import sys

print("=" * 70)
print("Testing Fixed Models")
print("=" * 70)

# Test 1: CB-QCCF
print("\n[1/4] Testing CB-QCCF...")
try:
    from src.models.fusion.class_balanced_quantum import ClassBalancedQuantumClassicalFusion
    
    model = ClassBalancedQuantumClassicalFusion(num_classes=2)
    dummy_input = torch.randn(2, 3, 224, 224)
    
    # Test training mode (should return single tensor)
    output = model(dummy_input)
    assert isinstance(output, torch.Tensor), "Should return single tensor"
    assert output.shape == (2, 2), f"Wrong shape: {output.shape}"
    
    # Test analysis mode (should return tuple)
    output_all = model(dummy_input, return_all=True)
    assert isinstance(output_all, tuple), "Should return tuple"
    assert len(output_all) == 3, "Should return 3 items"
    
    print("  ✅ CB-QCCF: PASSED")
    print(f"     - Training output shape: {output.shape}")
    print(f"     - Analysis output: 3 tensors (final, sens, spec)")
    
except Exception as e:
    print(f"  ❌ CB-QCCF: FAILED - {e}")
    sys.exit(1)

# Test 2: MSQF
print("\n[2/4] Testing MSQF...")
try:
    from src.models.fusion.multi_scale_quantum import MultiScaleQuantumFusion
    
    model = MultiScaleQuantumFusion(num_classes=2)
    dummy_input = torch.randn(2, 3, 224, 224)
    
    # Test forward pass
    output, attn_weights = model(dummy_input)
    assert isinstance(output, torch.Tensor), "Should return tensor"
    assert output.shape == (2, 2), f"Wrong shape: {output.shape}"
    
    print("  ✅ MSQF: PASSED")
    print(f"     - Output shape: {output.shape}")
    print(f"     - Attention weights shape: {attn_weights.shape}")
    
except Exception as e:
    print(f"  ❌ MSQF: FAILED - {e}")
    sys.exit(1)

# Test 3: Swin-V2-Small
print("\n[3/4] Testing Swin-V2-Small...")
try:
    from src.models.transformer import get_swin_v2_small
    
    model = get_swin_v2_small(num_classes=2, pretrained=False)
    dummy_input = torch.randn(2, 3, 256, 256)
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (2, 2), f"Wrong shape: {output.shape}"
    
    # Test gradient clipping
    model.train()
    output = model(dummy_input)
    loss = output.sum()
    loss.backward()
    
    # Clip gradients
    model.clip_gradients(max_norm=1.0)
    
    # Test LR recommendation
    lr = model.get_recommended_lr()
    assert lr == 1e-5, f"Wrong LR: {lr}"
    
    print("  ✅ Swin-V2-Small: PASSED")
    print(f"     - Output shape: {output.shape}")
    print(f"     - Recommended LR: {lr}")
    print(f"     - Gradient clipping: Available")
    
except Exception as e:
    print(f"  ❌ Swin-V2-Small: FAILED - {e}")
    sys.exit(1)

# Test 4: Ensemble Distillation
print("\n[4/4] Testing Ensemble Distillation...")
try:
    from src.models.fusion.ensemble_distillation import get_ensemble_distillation
    
    model = get_ensemble_distillation(
        num_classes=2,
        student_model='triple_branch_fusion',
        teacher_models=['swin_small', 'hybrid_vit', 'efficientnet'],
    )
    
    dummy_input = torch.randn(2, 3, 224, 224)
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        student_out, ensemble_out = model(dummy_input)
    
    assert student_out.shape == (2, 2), f"Wrong shape: {student_out.shape}"
    assert ensemble_out.shape == (2, 2), f"Wrong shape: {ensemble_out.shape}"
    
    print("  ✅ Ensemble Distillation: PASSED")
    print(f"     - Student output: {student_out.shape}")
    print(f"     - Ensemble output: {ensemble_out.shape}")
    
except Exception as e:
    print(f"  ❌ Ensemble Distillation: FAILED - {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print("\nAll models are ready for training. You can now run:")
print("  python run_pipeline.py --models cb_qccf multi_scale_quantum_fusion")
print("\nOr for all models including the previously crashed ones:")
print("  python run_pipeline.py --models cb_qccf multi_scale_quantum_fusion swin_v2_small ensemble_distillation")
print("=" * 70)
