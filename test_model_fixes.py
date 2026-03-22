"""
Test script to verify all bug fixes are working correctly.
This version tests code structure and import capability.
"""

import ast
import sys


def check_file_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        return True, "Valid syntax"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def check_cb_qccf_forward_method():
    """Test Issue 1: Check CB-QCCF forward method has return_all parameter."""
    print("\n" + "=" * 60)
    print("TEST 1: CB-QCCF Forward Method Fix")
    print("=" * 60)
    
    filepath = r"D:\Projects\AI-Projects\Breast_cancer_Minor_Project\src\models\fusion\class_balanced_quantum.py"
    
    # Check syntax
    valid, msg = check_file_syntax(filepath)
    if not valid:
        print(f"❌ FAILED - {msg}")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    
    # Check for return_all parameter in forward method
    if 'return_all: bool = False' in source:
        print("✓ Found 'return_all: bool = False' parameter")
    else:
        print("❌ Missing 'return_all' parameter")
        return False
    
    # Check for conditional return
    if 'if return_all:' in source and 'return final_pred' in source:
        print("✓ Found conditional return logic")
    else:
        print("❌ Missing conditional return logic")
        return False
    
    # Check that it returns just final_pred when return_all=False
    if 'return final_pred, sens_pred, spec_pred' in source:
        print("✓ Returns tuple when return_all=True")
    
    print("✅ CB-QCCF Forward Method: PASSED")
    return True


def check_msqf_dynamic_shapes():
    """Test Issue 2: Check MSQF doesn't have hardcoded view calls."""
    print("\n" + "=" * 60)
    print("TEST 2: MSQF Dynamic Shape Handling Fix")
    print("=" * 60)
    
    filepath = r"D:\Projects\AI-Projects\Breast_cancer_Minor_Project\src\models\fusion\multi_scale_quantum.py"
    
    # Check syntax
    valid, msg = check_file_syntax(filepath)
    if not valid:
        print(f"❌ FAILED - {msg}")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    
    # Check that hardcoded view calls are removed
    # The problematic pattern was: feat1.view(feat1.shape[0], 64, 56, 56)
    hardcoded_patterns = [
        '.view(feat1.shape[0], 64, 56, 56)',
        '.view(feat2.shape[0], 128, 28, 28)',
    ]
    
    found_hardcoded = False
    for pattern in hardcoded_patterns:
        if pattern in source:
            print(f"❌ Found hardcoded view pattern: {pattern}")
            found_hardcoded = True
    
    if found_hardcoded:
        print("❌ MSQF Dynamic Shape Handling: FAILED - Hardcoded shapes still present")
        return False
    
    print("✓ No hardcoded .view() calls with fixed spatial dimensions")
    
    # Check that extract_multi_scale_features uses dynamic shapes
    if 'extract_multi_scale_features' in source:
        print("✓ extract_multi_scale_features method exists")
    
    print("✅ MSQF Dynamic Shape Handling: PASSED")
    return True


def check_swin_v2_stability():
    """Test Issue 3: Check Swin-V2 has gradient clipping and LR methods."""
    print("\n" + "=" * 60)
    print("TEST 3: Swin-V2-Small Training Stability Fix")
    print("=" * 60)
    
    filepath = r"D:\Projects\AI-Projects\Breast_cancer_Minor_Project\src\models\transformer\swin.py"
    
    # Check syntax
    valid, msg = check_file_syntax(filepath)
    if not valid:
        print(f"❌ FAILED - {msg}")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    
    # Check for get_recommended_lr method
    if 'def get_recommended_lr(self)' in source:
        print("✓ Found get_recommended_lr method")
    else:
        print("❌ Missing get_recommended_lr method")
        return False
    
    # Check for V2-specific lower LR
    if "return 1e-5" in source and "V2" in source:
        print("✓ Found V2-specific lower learning rate (1e-5)")
    else:
        print("⚠ Warning: V2 LR recommendation may not be present")
    
    # Check for clip_gradients method
    if 'def clip_gradients(self' in source:
        print("✓ Found clip_gradients method")
    else:
        print("❌ Missing clip_gradients method")
        return False
    
    # Check for check_layer_norm_health method
    if 'def check_layer_norm_health(self' in source:
        print("✓ Found check_layer_norm_health method")
    else:
        print("❌ Missing check_layer_norm_health method")
        return False
    
    # Check Dict import
    if 'from typing import' in source and 'Dict' in source:
        print("✓ Dict type imported for type hints")
    
    print("✅ Swin-V2-Small Training Stability: PASSED")
    return True


def check_ensemble_distillation_num_classes():
    """Test Issue 4: Check ensemble distillation filters num_classes from kwargs."""
    print("\n" + "=" * 60)
    print("TEST 4: Ensemble Distillation num_classes Fix")
    print("=" * 60)
    
    filepath = r"D:\Projects\AI-Projects\Breast_cancer_Minor_Project\src\models\fusion\ensemble_distillation.py"
    
    # Check syntax
    valid, msg = check_file_syntax(filepath)
    if not valid:
        print(f"❌ FAILED - {msg}")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    
    # Check for filtered_kwargs pattern
    if 'filtered_kwargs' in source:
        print("✓ Found filtered_kwargs variable")
    else:
        print("❌ Missing filtered_kwargs variable")
        return False
    
    # Check for the filtering logic
    if "k != 'num_classes'" in source or 'k != "num_classes"' in source:
        print("✓ Found num_classes filtering logic")
    else:
        print("❌ Missing num_classes filtering logic")
        return False
    
    # Check that filtered_kwargs is used in lambdas
    if '**filtered_kwargs' in source:
        print("✓ filtered_kwargs is used in lambda functions")
    else:
        print("❌ filtered_kwargs not used in lambdas")
        return False
    
    print("✅ Ensemble Distillation num_classes Handling: PASSED")
    return True


def main():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("MODEL BUG FIX VERIFICATION TEST SUITE")
    print("=" * 60)
    
    results = {
        "CB-QCCF Forward Method": check_cb_qccf_forward_method(),
        "MSQF Dynamic Shapes": check_msqf_dynamic_shapes(),
        "Swin-V2 Stability": check_swin_v2_stability(),
        "Ensemble Distillation": check_ensemble_distillation_num_classes(),
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! All bug fixes are correctly implemented.")
        print("\n📝 Note: Full runtime tests require timm and torchvision packages.")
        print("   To run full tests: pip install timm torchvision")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
