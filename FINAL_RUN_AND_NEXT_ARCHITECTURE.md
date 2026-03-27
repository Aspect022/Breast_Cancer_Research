# Final Run Plan

## Immediate Phase: Clean Official Rerun

Use `config_final.yaml` for the official rerun set:

- `efficientnet`
- `hybrid_vit`
- `swin_small`
- `convnext_small`
- `dual_branch_fusion`
- `qenn_u3`
- `quantum_enhanced_fusion`
- `triple_branch_fusion`
- `triple_branch_fusion_quantum`
- `triple_branch_fusion_bottleneck`
- `triple_branch_fusion_cnn_featuremap_quantum`
- `triple_branch_fusion_vit_featuremap_quantum`
- `cb_qccf`
- `cb_qccf_convnet_efficient`
- `cb_qccf_swin_convnet`
- `multi_scale_quantum_fusion`

Rules for this phase:

- No PennyLane path
- Local GPU-native quantum only
- Clean W&B project: `breast-cancer-final`
- Focus on validation and test metrics that matter for the paper:
  `accuracy`, `auc`, `f1`, `mcc`, `sensitivity`, `specificity`, `fnr`, `ppv`, `npv`, `balanced_accuracy`

## Next Phase: Feature-Map Quantum Fusion

Requested direction:

1. CNN or ViT feature extractor
2. Feature extractor outputs a feature map
3. Quantum block operates on the feature map representation
4. Three parallel branches continue
5. Existing fusion pipeline continues after that

Recommended implementation shape:

```text
Input
  -> Shared backbone or branch-specific backbone
  -> Intermediate feature map
  -> Spatial pooling / token pooling
  -> Quantum projector (feature map -> qubit angles)
  -> Local GPU quantum circuit
  -> Quantum-enhanced branch embedding
  -> Three-branch fusion
  -> Cross-attention / weighted fusion / classifier
```

Recommended first prototype:

- Add quantum after projected branch features, not raw 2D maps
- Start with one branch only
- Use `u3 + cyclic` first, since it is the strongest current local quantum setting
- Compare against TBCA classical and TBCA quantum bottleneck

## Accuracy Reality Check

Reaching `94-95%` on BreakHis may not come from architecture alone.
The highest leverage items are likely:

- stronger preprocessing and stain normalization
- magnification-aware training
- patient-level split auditing
- class balancing and threshold tuning
- test-time augmentation
- targeted hyperparameter sweeps on the best 3-4 models

Architecture can still help, but data protocol and optimization will likely matter just as much.
