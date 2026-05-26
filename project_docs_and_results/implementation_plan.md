# 🚀 Implementation Plan: Transformers + Quantum Focus

**Updated:** 2026-03-19  
**Timeline:** 10 weeks  
**Goal:** ≥88% accuracy, <2 hours quantum training time

---

## 📋 Executive Summary

**Strategic Shift:**
- ❌ Remove SNN from project focus (clinically unusable - 62.8% FNR)
- ✅ Double down on **Transformers + Quantum**
- ✅ Primary contribution: **Dual-branch dynamic fusion** (Swin + ConvNeXt)
- ✅ Secondary contribution: **GPU-accelerated quantum** (100× speedup target)

**Target Venue:** MICCAI 2026 (March deadline) or ISBI 2026 (October deadline)

---

## 🗓️ Phase Timeline

### Phase 1 (Week 1-2): Quick Wins + Foundation

**Goal:** Establish new baseline transformers + prepare quantum migration

| Step | Task | Effort | Dependencies | Completion Criteria |
|------|------|--------|--------------|---------------------|
| 1.1 | Install dependencies (timm, torchquantum) | Low | None | `import timm`, `import torchquantum` work |
| 1.2 | Implement Swin Transformer Tiny (standalone) | Medium | None | Swin-T achieves ≥83% accuracy on single fold |
| 1.3 | Implement ConvNeXt Tiny (standalone) | Medium | None | ConvNeXt-T achieves ≥83% accuracy on single fold |
| 1.4 | Create feature extraction utility | Low | 1.2, 1.3 | Can extract features from both models |
| 1.5 | Benchmark current quantum timing (PennyLane) | Low | None | Document exact timing breakdown |
| 1.6 | Implement vectorized quantum circuit (RY-only) | High | None | Single forward pass <10ms (vs ~500ms current) |

**Milestone 1 (End of Week 2):**
- ✅ Swin-T and ConvNeXt-T trained and validated (single fold)
- ✅ Vectorized quantum circuit working on GPU
- ✅ Timing benchmark: quantum forward <10ms

**Deliverables:**
- `src/models/transformer/swin.py` - Swin wrapper
- `src/models/transformer/convnext.py` - ConvNeXt wrapper
- `src/models/quantum/vectorized_circuit.py` - Custom GPU quantum
- Single-fold results for Swin-T, ConvNeXt-T

---

### Phase 2 (Week 3-5): Core New Architectures

**Goal:** Implement full transformer zoo + dual-branch fusion

| Step | Task | Effort | Dependencies | Completion Criteria |
|------|------|--------|--------------|---------------------|
| 2.1 | Implement Swin-Small and Swin-V2 variants | Medium | 1.2 | Swin-S achieves ≥85% accuracy |
| 2.2 | Implement ConvNeXt Small | Low | 1.3 | ConvNeXt-S achieves ≥84% accuracy |
| 2.3 | Implement DeiT-Tiny/Small/Base | Medium | None | DeiT-S achieves ≥82% accuracy |
| 2.4 | Implement feature alignment module | Medium | 1.4 | Features can be concatenated without shape mismatch |
| 2.5 | Implement gating network (Conv3x3→Sigmoid) | High | 2.4 | Gate produces meaningful alpha values (0.2-0.8 range) |
| 2.6 | Implement dual-branch fusion model | High | 2.1, 2.2, 2.5 | Full model trains without NaN/gradient issues |
| 2.7 | Add entropy regularization to loss | Low | 2.6 | Loss function includes entropy term |

**Milestone 2 (End of Week 5):**
- ✅ Transformer zoo complete (Swin-T/S/V2, ConvNeXt-T/S, DeiT-T/S/B)
- ✅ Dual-branch fusion model training successfully
- ✅ Fusion model achieves ≥87% accuracy (beats 85% baseline)

**Deliverables:**
- `src/models/transformer/deit.py` - DeiT wrapper
- `src/models/fusion/dual_branch.py` - Swin+ConvNeXt fusion
- `src/models/fusion/gating.py` - Gating network
- 5-fold CV results for all standalone transformers
- 5-fold CV results for fusion model

---

### Phase 3 (Week 6-8): Advanced Fusion + Quantum

**Goal:** Integrate quantum into fusion + rotation gate ablation

| Step | Task | Effort | Dependencies | Completion Criteria |
|------|------|--------|--------------|---------------------|
| 3.1 | Migrate quantum to TorchQuantum (if better than custom) | Medium | 1.6 | TorchQuantum circuit matches custom timing |
| 3.2 | Implement rotation gate variants (RY, RY+RZ, U3, RX+RY+RZ) | High | 1.6 | All 4 variants implemented and testable |
| 3.3 | Integrate quantum layer into fusion model | High | 2.6, 3.2 | Quantum-enhanced fusion trains successfully |
| 3.4 | Test entanglement strategies (cyclic CNOT vs alternatives) | Medium | 3.2 | Compare 2-3 entanglement patterns |
| 3.5 | Quantum ablation study (n_qubits, n_layers) | Medium | 3.3 | Identify optimal quantum configuration |
| 3.6 | Implement CrossViT or Med-Former | High | None | Medical-specific transformer achieves ≥84% |
| 3.7 | Full model comparison (all models × 5-fold CV) | Medium | All previous | Complete comparison table generated |

**Milestone 3 (End of Week 8):**
- ✅ Quantum rotation gate ablation complete
- ✅ Best quantum config identified: ___ rotation, ___ entanglement, ___ qubits
- ✅ Quantum-enhanced fusion achieves ≥88% accuracy
- ✅ Training time: <2 hours per 5-fold CV (target: <10× slowdown)

**Deliverables:**
- `src/models/quantum/rotation_variants.py` - RY, RY+RZ, U3, RX+RY+RZ circuits
- `src/models/fusion/quantum_fusion.py` - Quantum-enhanced fusion
- `src/models/transformer/medical.py` - CrossViT/Med-Former
- Quantum ablation study results
- Complete comparison table

---

### Phase 4 (Week 9-10): Ablation Studies + Polish

**Goal:** Comprehensive analysis + publication preparation

| Step | Task | Effort | Dependencies | Completion Criteria |
|------|------|--------|--------------|---------------------|
| 4.1 | Statistical significance testing (paired t-tests) | Low | 3.7 | P-values for all model comparisons |
| 4.2 | Ablation: fusion components (gate, normalization) | Medium | 2.6 | Quantify contribution of each fusion component |
| 4.3 | Ablation: transformer backbones (Swin vs ConvNeXt) | Low | 3.7 | Identify best backbone combination |
| 4.4 | Interpretability: Grad-CAM, attention visualization | Medium | 3.7 | Generate attention maps for all models |
| 4.5 | External validation (optional: Camelyon16 subset) | High | 3.7 | Test generalization on external dataset |
| 4.6 | Create comparison figures/tables for paper | Medium | 4.1-4.5 | Publication-ready results section |
| 4.7 | Write technical report / paper draft | High | 4.6 | Complete draft ready for review |

**Milestone 4 (End of Week 10):**
- ✅ Complete ablation study table
- ✅ Statistical significance analysis (p-values)
- ✅ Interpretability visualizations (Grad-CAM, attention maps)
- ✅ Paper draft / technical report complete

**Deliverables:**
- `outputs/ablation_studies/` - All ablation results
- `outputs/visualizations/` - Grad-CAM, attention maps
- `docs/paper_draft.md` - Initial paper draft
- `docs/results_summary.md` - Complete results summary

---

## 📊 Success Metrics

### Accuracy Targets

| Model | Current | Target | Stretch Goal |
|-------|---------|--------|--------------|
| EfficientNet-B3 (baseline) | 85% | 85% | 85% |
| Swin Transformer Tiny | — | 83% | 85% |
| Swin Transformer Small | — | 85% | 87% |
| Swin Transformer V2 | — | 86% | 88% |
| ConvNeXt Tiny | — | 83% | 85% |
| ConvNeXt Small | — | 84% | 86% |
| DeiT Small | — | 82% | 84% |
| **Dual-Branch Fusion (Swin+ConvNeXt)** | — | **87%** | **89%** |
| CNN+PQC (current, PennyLane) | 84% | 84% | 85% |
| **Quantum-Enhanced Fusion** | — | **88%** | **90%** |
| CrossViT / Med-Former | — | 84% | 86% |

### Training Time Targets (5-fold CV)

| Model | Current (PennyLane) | Target | Speedup |
|-------|---------------------|--------|---------|
| EfficientNet-B3 | ~10 min | ~10 min | 1× |
| CNN+PQC | 48,284s (13.4 hrs) | <2 hrs | **6-7×** |
| Vectorized Quantum (custom) | — | <500s | **100×** |
| Swin Transformer Tiny | — | <30 min | — |
| ConvNeXt Tiny | — | <25 min | — |
| Dual-Branch Fusion | — | <45 min | — |

### Research Output Goals

1. **Comparison Table:** 10 models × 5 metrics (Accuracy, AUC, Sensitivity, Specificity, F1, Training Time)
2. **Ablation Studies:**
   - Quantum rotation gates (4 variants) × entanglement strategies (2-3 variants)
   - Fusion components (gate, normalization, refinement)
   - Transformer backbones (Swin vs ConvNeXt vs DeiT)
3. **Statistical Analysis:** P-values for all pairwise comparisons
4. **Interpretability:** Grad-CAM heatmaps, attention visualizations, gate distributions
5. **Publication:**
   - Minimum: Workshop paper (fusion model + quantum ablation)
   - Target: MICCAI/ISBI conference paper
   - Stretch: Journal article (MedIA, IEEE TMI)

---

## ⚠️ Risk Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **TorchQuantum doesn't work on Windows** | Medium | High | **Fallback:** Use custom vectorized implementation (proven in EEG project, 50-100× speedup) |
| **Fusion doesn't improve accuracy** | Medium | Medium | **Fallback:** Use ensemble averaging instead of learned fusion; Swin/ConvNeXt ensemble likely beats 85% |
| **Quantum still too slow after migration** | Low | High | **Fallback:** Reduce qubit count (8→4), reduce layers (2→1), use quantum only for final classification |
| **GPU VRAM insufficient for fusion** | Medium | Medium | **Fallback:** Gradient accumulation, smaller batch size (16→8), mixed precision training |
| **Gating network collapses (alpha=0 or 1)** | Medium | Medium | **Fallback:** Add entropy regularization, constrain gate range, use temperature scaling |

### Timeline Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Phase 1 takes >2 weeks** | Low | High | Prioritize Swin-T + quantum migration only; defer ConvNeXt to Phase 2 |
| **Quantum ablation consumes Phase 3** | Medium | Medium | Set hard deadline; if not working by Week 7, fall back to classical fusion only |
| **External validation fails** | High | Low | Make Phase 4.5 optional; BreakHis-only results still publishable |

---

## 🔧 Technical Specifications

### 1. Project Structure

```
src/
├── data/
│   └── dataset.py              # BreakHis loader + GroupKFold
├── models/
│   ├── transformer/
│   │   ├── swin.py             # Swin-T/S/V2 wrappers
│   │   ├── convnext.py         # ConvNeXt-T/S wrappers
│   │   ├── deit.py             # DeiT-T/S/B wrappers
│   │   └── medical.py          # CrossViT, Med-Former
│   ├── quantum/
│   │   ├── vectorized_circuit.py  # Custom GPU quantum
│   │   ├── rotation_variants.py   # RY, RY+RZ, U3, RX+RY+RZ
│   │   └── hybrid_quantum.py      # CNN+PQC, Fusion+PQC
│   └── fusion/
│       ├── dual_branch.py      # Swin+ConvNeXt fusion
│       ├── gating.py           # Gating network
│       └── quantum_fusion.py   # Quantum-enhanced fusion
└── utils/
    └── metrics.py              # Metrics + plotting
```

### 2. Dual-Branch Fusion Architecture

```python
class DualBranchDynamicFusion(nn.Module):
    """
    Dual-Branch Dynamic Fusion Model.
    
    Input: (B, 3, 224, 224) → ImageNet normalization
    
    Branch A (Global): Swin Transformer Tiny → F_sw
    Branch B (Local): ConvNeXt Tiny → F_cn
    
    Feature Alignment:
        - Channel-first: (B, C, H, W)
        - Bilinear resize to match spatial dims
        - BatchNorm on both branches
    
    Fusion Prep:
        - Concatenate channels: F_cat = concat(F_sw, F_cn)
    
    Gating Network:
        - Conv3x3 → BN → ReLU
        - Conv1x1 → Sigmoid → alpha (B, 1, H, W)
    
    Dynamic Fusion:
        - F_fused = alpha * F_cn + (1-alpha) * F_sw
    
    Refinement:
        - Conv3x3 → BN → ReLU → Dropout
    
    Classification:
        - Global Average Pool
        - FC(512) → ReLU → Dropout
        - FC(256) → ReLU → Dropout
        - FC(num_classes)
    
    Training:
        - AdamW(lr=2e-5, weight_decay=1e-4)
        - CosineAnnealingWarmRestarts(T_0=10, T_mult=2)
        - Loss: CrossEntropy + 0.01 * entropy_regularization(alpha)
    """
```

### 3. Quantum Circuit Template (Vectorized)

```python
class VectorizedQuantumCircuit(nn.Module):
    def __init__(self, n_qubits=8, n_layers=2, rotation_axes='y'):
        super().__init__()
        self.n_qubits = n_qubits
        self.state_dim = 2 ** n_qubits
        self.rotation_axes = rotation_axes
        
        # Trainable parameters
        self.params = nn.ParameterDict()
        for i, axis in enumerate(rotation_axes):
            self.params[f'theta_{i}_{axis}'] = nn.Parameter(
                torch.randn(n_layers, n_qubits) * 0.1
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_qubits) - angle encoded features
        batch_size = x.shape[0]
        
        # Initialize |0⟩^⊗n state
        state = torch.zeros(batch_size, self.state_dim, 
                           dtype=torch.complex64, device=x.device)
        state[:, 0] = 1.0 + 0.0j
        
        # Angle encoding (RY on each qubit)
        for i in range(self.n_qubits):
            state = self._apply_ry_gate(state, x[:, i], i)
        
        # Variational layers
        for layer in range(self.n_layers):
            for i, axis in enumerate(self.rotation_axes):
                for qubit in range(self.n_qubits):
                    angle = self.params[f'theta_{i}_{axis}'][layer, qubit]
                    state = self._apply_rotation_gate(state, angle, qubit, axis)
            
            # CNOT ring entanglement
            state = self._apply_cnot_ring(state)
        
        return state  # (batch, state_dim)
    
    def measure(self, state: torch.Tensor) -> torch.Tensor:
        # Pauli-Z measurement on all qubits
        expectations = []
        for i in range(self.n_qubits):
            obs = self._create_pauli_z_observable(i)
            exp = torch.real(torch.sum(state.conj() * obs @ state.T, dim=1))
            expectations.append(exp)
        return torch.stack(expectations, dim=1)  # (batch, n_qubits)
```

### 4. Training Configuration (config.yaml)

```yaml
models:
  # Standalone transformers
  swin_tiny:
    enabled: true
    batch_size: 32
    lr: 2.0e-5
    weight_decay: 1.0e-4
    use_amp: true
    variant: "tiny"  # tiny, small, v2
  
  convnext_tiny:
    enabled: true
    batch_size: 32
    lr: 2.0e-5
    weight_decay: 1.0e-4
    use_amp: true
    variant: "tiny"  # tiny, small
  
  deit_small:
    enabled: true
    batch_size: 32
    lr: 2.0e-5
    weight_decay: 1.0e-4
    use_amp: true
    variant: "small"  # tiny, small, base
  
  # Fusion models
  dual_branch_fusion:
    enabled: true
    batch_size: 16  # Smaller due to two backbones
    lr: 2.0e-5
    weight_decay: 1.0e-4
    use_amp: true
    swin_variant: "tiny"
    convnext_variant: "tiny"
    dropout: 0.3
    entropy_weight: 0.01
    scheduler: "CosineAnnealingWarmRestarts"
    T_0: 10
    T_mult: 2
  
  # Quantum models
  quantum_enhanced_fusion:
    enabled: true
    batch_size: 16
    lr: 2.0e-5
    weight_decay: 1.0e-4
    use_amp: false  # Quantum may not support AMP
    n_qubits: 8
    n_layers: 2
    rotation_config: "ry_only"  # ry_only, ry_rz, u3, rx_ry_rz
    entanglement: "cyclic"  # cyclic, full, tree
    use_torchquantum: false  # false = custom vectorized
```

---

## 🚀 Next Immediate Actions

### Right Now (Today):

1. **Install dependencies:**
   ```bash
   pip install timm
   # Try TorchQuantum first
   git clone https://github.com/mit-han-lab/torchquantum.git
   cd torchquantum
   pip install --editable .
   ```

2. **Create new project structure:**
   ```bash
   mkdir -p src/models/transformer
   mkdir -p src/models/quantum
   mkdir -p src/models/fusion
   ```

3. **Reference EEG project quantum implementation:**
   - Navigate to: `D:\Projects\AI-Projects\EEG`
   - Copy quantum circuit implementation
   - Adapt for 2D image features (currently designed for 1D EEG)

4. **Start Swin-T implementation:**
   ```python
   from timm import create_model
   model = create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=2)
   ```

### This Week (Week 1):

- [ ] Vectorized quantum circuit working on GPU
- [ ] Swin-T achieving ≥80% accuracy on single fold
- [ ] ConvNeXt-T achieving ≥80% accuracy on single fold
- [ ] Timing benchmark comparing PennyLane vs vectorized

### Week 2 Checkpoint:

- [ ] 5-fold CV results for Swin-T and ConvNeXt-T
- [ ] Decision: TorchQuantum vs custom vectorized
- [ ] Feature extraction utility working
- [ ] Begin fusion model implementation

---

## 📚 Related Documents

- **Issues Tracker:** `issues.md` - All identified issues and limitations
- **Research Prompt:** `research_prompt.md` - Comprehensive research questions
- **Original Plan:** `plan.md` - Original paradigm comparison design

---

## 📞 Contact & Support

For questions about:
- **Quantum implementation:** Reference EEG project at `D:\Projects\AI-Projects\EEG`
- **Transformer architectures:** See `research_prompt.md` Task 1
- **Fusion design:** See `research_prompt.md` Task 3
- **Publication strategy:** See `research_prompt.md` Task 5C

---

**Good luck! 🎯**
