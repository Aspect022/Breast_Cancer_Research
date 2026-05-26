# TBCA Quantum Placement Analysis + New Variants

---

## The Quantum Placement Question

The user correctly identified a key architectural insight. Let's trace through it carefully.

---

## Current Quantum Placement (Published Results)

Looking at the four existing TBCA variants:

```
Variant A: TBCA-Quantum-Fusion
  [Swin, ConvNeXt, EfficientNet] → project → cross-attention → weighted-sum
  → [FUSED 768d] → VQC → classifier
  AUC: 99.01%, Acc: 91.03%, FNR: 0.31%
  ❌ Problem: quantum operates on ALREADY-COLLAPSED single vector
  
Variant B: TBCA-Quantum-Bottleneck  
  [Swin, ConvNeXt] → VQC each → project → cross-attention → weighted-sum
  → classifier  (EfficientNet skipped for efficiency)
  AUC: 97.24%, Acc: 92.94%, FNR: 1.36%
  ❌ Problem: quantum only on 2 branches, after projection but before cross-attn

Variant C: TBCA-CNN-FM-Quantum [BEST CURRENT MODEL]
  [Swin, ConvNeXt] + [EfficientNet spatial map → VQC → enriched effnet feature]
  → cross-attention → weighted-sum → classifier
  AUC: 98.92%, Acc: 95.48%, FNR: 0.94%
  ✅ VQC operates on EfficientNet's INTERMEDIATE SPATIAL FEATURE MAP
     = rich local spatial structure, before GAP collapses it

Variant D: TBCA-ViT-FM-Quantum [ABLATION]
  Same as C but ViT patch tokens instead of CNN feature map into VQC
  AUC: 91.08%, Acc: 91.88%, FNR: 7.58%
  ❌ ViT's global attention tokens lose local structure needed for VQC compression
```

**Key finding from ablation (already in paper):**
> "CNN intermediate maps retain local structural gradients on nuclear boundaries, gland edges, staining textures that remain informationally dense when compressed to 8 values."

---

## The User's Insight: What's Missing?

Variant C puts quantum **inside Branch 3** before any cross-branch interaction. The VQC only sees EfficientNet's local multi-scale features.

**What if the quantum circuit could see the result of cross-branch interaction?**

After cross-attention, each branch's features have been enriched by the other two branches:
- `swin_enhanced` = Swin context + ConvNeXt texture signal + EfficientNet morphology
- `convnext_enhanced` = ConvNeXt texture + Swin context signal
- `effnet_enhanced` = EfficientNet morphology + Swin context signal

**These enhanced features carry CROSS-BACKBONE INFORMATION that the current VQC never sees.**

A quantum circuit operating on cross-attention-enhanced features can potentially:
1. Find quantum correlations between GLOBAL context + LOCAL texture + MULTI-SCALE morphology
2. Operate in a richer feature space than any single backbone can provide
3. Capture inter-backbone relationships in the Hilbert space (not possible classically without exponential parameters)

---

## Four New TBCA Variants

### Variant E: TBCA-PostCrossAttn-Quantum ⭐ [MOST PROMISING]

**Place VQC AFTER cross-attention, operating on the combined enhanced features**

```
[Swin, ConvNeXt, EfficientNet]
    ↓
[project each → 768d]
    ↓
Cross-attention enhancement:
  swin_enh     = f(swin, convnext, effnet)    [B, 768]
  convnext_enh = f(convnext, swin)             [B, 768]
  effnet_enh   = f(effnet, swin)               [B, 768]
    ↓
Stack enhanced features:
  combined = stack([swin_enh, convnext_enh, effnet_enh])   [B, 3, 768]
    ↓
VQC Branch (parallel to classical branch):
  Linear(3×768, 8) → tanh → z' [B, 8]
  ↓
  Amplitude encode → 8-qubit state
  ↓
  2× (U3 rotations + cyclic CNOT ring)
  ↓
  Measure ⟨Z_i⟩ → [B, 8]
  Linear(8, 768) → f_q [B, 768]

Classical branch (parallel):
  Weighted sum of enhanced features → f_classical [B, 768]

Residual fusion:
  f* = f_classical + f_q    [B, 768]    ← analogous to TBCA Eq. 6
    ↓
Self-attention refinement → Classifier head
```

**Why this should outperform CNN-FM-Quantum:**

| Dimension | CNN-FM-Quantum (C) | PostCrossAttn-Quantum (E) |
|---|---|---|
| VQC input | EfficientNet spatial map (single backbone) | Combined enhanced features (all 3 backbones) |
| Cross-branch info | None (pre-interaction) | Full (post-interaction) |
| Biological richness | Local CNN features | Global + texture + multi-scale |
| Compression | CNN spatial → 8 values | Cross-backbone fused → 8 values |
| Hilbert encoding | Single backbone's feature space | Multi-backbone integrated space |

**Implementation (changes to existing code):**
```python
def forward(self, x):
    # Steps 1-4: same as existing TBCA
    swin_feat, convnext_feat, effnet_feat = self.extract_features(x)
    swin_proj, convnext_proj, effnet_proj = self.project_features(...)
    swin_enh, convnext_enh, effnet_enh = self.apply_cross_attention(...)
    
    # NEW: Stack enhanced features → VQC
    combined = torch.cat([swin_enh, convnext_enh, effnet_enh], dim=-1)  # [B, 2304]
    z_prime = torch.tanh(self.quantum_input_proj(combined))              # [B, 8]
    f_q = self.vqc_branch(z_prime)                                       # [B, 768]
    
    # Classical path (existing)
    f_classical = self.fuse_features(swin_enh, convnext_enh, effnet_enh) # [B, 768]
    
    # Residual (same as Eq. 6 in paper)
    fused = f_classical + f_q                                             # [B, 768]
    
    # Steps 7-8: same
    fused_refined = self.fusion_attention(fused)
    return self.classifier(fused_refined)
```

**New modules needed:**
```python
self.quantum_input_proj = nn.Linear(768 * 3, 8)  # 2304 → 8
self.vqc_branch = QuantumVQCBranch(n_qubits=8, n_layers=2, output_dim=768)
# QuantumVQCBranch: same PennyLane 8-qubit VQC as in CNN-FM-Quantum paper
```

---

### Variant F: TBCA-DualQuantum ⭐⭐ [NOVEL CONTRIBUTION]

**Apply VQC TWICE: once on CNN feature map (like current best), AND once on post-cross-attention features**

```
[Swin, ConvNeXt, EfficientNet_backbone]
    ↓
EfficientNet spatial feature map:
  effnet_map [B, 2048, H, W]
  ↓
  VQC-1 (CNN-FM): same as current best → f_q1 [B, 768]  (TBCA Eq. 6)
  effnet_feat* = effnet_pool + f_q1
    ↓
Cross-attention with enriched EfficientNet:
  swin_enh, convnext_enh, effnet_enh
    ↓
VQC-2 (PostCrossAttn): combined enhanced features → f_q2 [B, 768]
  fused = f_classical + f_q2

Self-attention → Classifier
```

**Architecture has TWO quantum stages:**
- VQC-1 enriches one branch BEFORE interaction (local spatial features)
- VQC-2 enriches the FUSED representation AFTER interaction (cross-branch features)

This is a genuine dual quantum architecture. Total params: ~142.5M + 48 quantum params (VQC-1) + 48 more (VQC-2) = still mostly classical params, quantum is lightweight.

---

### Variant G: TBCA-QuantumCrossAttention [MOST NOVEL — HIGH RISK/REWARD]

**Replace the classical cross-attention MODULE ITSELF with a quantum kernel attention mechanism**

```
For any two branches i and j:
  Classical CrossAttention: softmax(Q·K^T / √d) · V
  
  Quantum CrossAttention (new):
    Encode (F_i, F_j) into 8-qubit state
    VQC computes quantum correlations
    Measurements produce attention weights
    Apply to values
```

**Concrete design:**
```python
class QuantumCrossAttention(nn.Module):
    """Replace classical cross-attention with VQC-based attention"""
    def __init__(self, dim=768, n_qubits=8):
        super().__init__()
        self.q_compress = nn.Linear(dim, n_qubits // 2)  # 4 qubits for query
        self.k_compress = nn.Linear(dim, n_qubits // 2)  # 4 qubits for key
        self.vqc = VQCLayer(n_qubits=n_qubits, n_layers=2)
        self.attn_expand = nn.Linear(n_qubits, dim)
        self.v_proj = nn.Linear(dim, dim)
        
    def forward(self, query, key, value):
        # Compress query and key to half the qubit budget each
        q_enc = torch.tanh(self.q_compress(query))  # [B, 4]
        k_enc = torch.tanh(self.k_compress(key))    # [B, 4]
        
        # Concatenate for joint quantum encoding
        qk_joint = torch.cat([q_enc, k_enc], dim=-1)  # [B, 8]
        
        # VQC computes correlations in Hilbert space
        qk_quantum = self.vqc(qk_joint)  # [B, 8]
        
        # Attention weights from quantum measurement
        attn_weights = torch.sigmoid(self.attn_expand(qk_quantum))  # [B, dim]
        
        # Apply to values
        v = self.v_proj(value)
        attended = attn_weights * v
        
        return self.layer_norm(query + attended)
```

**Risk:** May destabilize training (quantum attention is not PSD in the standard softmax sense). Use residual connections and careful learning rate.

---

### Variant H: TBCA-AllBranch-FM-Quantum [SYSTEMATIC ABLATION]

**Apply VQC to ALL three branches' intermediate feature maps (not just EfficientNet)**

```
Branch 1 (Swin):
  swin_tokens [B, N, 768] → quantum_swin [B, 768] 
  (compress tokens: global avg pool → 8 → VQC → 768)
  swin_feat* = swin_feat + quantum_swin

Branch 2 (ConvNeXt):
  convnext_feature_map [B, 768, H, W] → quantum_convnext [B, 768]
  (adaptive pool → 8 → VQC → 768)
  convnext_feat* = convnext_feat + quantum_convnext

Branch 3 (EfficientNet): [same as current CNN-FM-Quantum]
  effnet_map [B, 2048, H, W] → quantum_effnet [B, 768]
  effnet_feat* = effnet_feat + quantum_effnet

→ Cross-attention of all enriched branches → fusion → classifier
```

**This systematically tests: does applying VQC to all three branches give additive improvement?**

Total VQC params: 3 × 48 = 144 quantum parameters (still tiny vs 141M classical)

---

## Summary: New Variants to Test

| Variant | VQC Placement | Input to VQC | Novelty | Expected |
|---|---|---|---|---|
| C (current best) | Inside Branch 3 | EfficientNet spatial map | Published | 95.48% |
| E: PostCrossAttn | After cross-attention | All 3 branches combined | **High** | **95.8–96.5%** |
| F: DualQuantum | Both C + E | Feature map AND combined | **Very High** | **95.5–96.3%** |
| G: QuantumCrossAttn | AS cross-attention | (Q, K) joint encoding | **Highest** | Unknown (risky) |
| H: AllBranch-FM | All 3 branch maps | Per-branch spatial maps | Medium | ~95.2–95.8% |

---

## Implementation Priority

1. **Start with Variant E (PostCrossAttn-Quantum)** — minimal code change, highest expected gain, clearest theoretical motivation
2. **Then F (DualQuantum)** — compound the two best ideas
3. **Then H (AllBranch)** — systematic ablation
4. **Then G (QuantumCrossAttn)** — last, highest risk/reward

---

## Code Changes Required

**Add to `src/models/fusion/triple_branch.py`:**

```python
class TripleBranchCrossAttention(nn.Module):
    def __init__(self, ..., quantum_mode='cnn_fm'):
        # quantum_mode options:
        #   'none'              → classical TBCA
        #   'cnn_fm'            → current best (Variant C)
        #   'post_cross_attn'   → Variant E (NEW)
        #   'dual_quantum'      → Variant F (NEW)
        #   'all_branch_fm'     → Variant H (NEW)
        #   'quantum_cross_attn'→ Variant G (NEW, risky)
        
        if quantum_mode in ['post_cross_attn', 'dual_quantum']:
            # VQC that takes combined enhanced features
            self.post_crossattn_quantum = PostCrossAttnVQC(
                input_dim=768 * 3,   # all three enhanced branches
                n_qubits=8,
                n_layers=2,
                output_dim=768
            )
        ...
    
    def forward(self, x):
        swin_feat, convnext_feat, effnet_feat = self.extract_features(x)
        
        # CNN-FM quantum on EfficientNet branch (Variant C path)
        if self.quantum_mode in ['cnn_fm', 'dual_quantum']:
            effnet_feat = effnet_feat + self.feature_map_quantum(effnet_map)
        
        swin_proj, convnext_proj, effnet_proj = self.project_features(...)
        swin_enh, convnext_enh, effnet_enh = self.apply_cross_attention(...)
        
        fused = self.fuse_features(swin_enh, convnext_enh, effnet_enh)
        
        # Post-cross-attn quantum (Variant E/F path)
        if self.quantum_mode in ['post_cross_attn', 'dual_quantum']:
            combined = torch.cat([swin_enh, convnext_enh, effnet_enh], dim=-1)
            f_q = self.post_crossattn_quantum(combined)
            fused = fused + f_q   # residual, same as Eq. 6
        
        fused_refined = self.fusion_attention(fused)
        return self.classifier(fused_refined)
```

---

## Config for New Variants

```yaml
# Add to config.yaml

triple_branch_fusion_post_crossattn_quantum:
  enabled: true
  batch_size: 8           # slightly smaller for memory
  lr: 1e-5
  weight_decay: 1e-4
  use_amp: true
  swin_variant: "small"
  convnext_variant: "small"
  efficientnet_variant: "b5"
  dropout: 0.3
  fusion_dim: 768
  num_heads: 8
  quantum_mode: "post_cross_attn"
  quantum_n_qubits: 8
  quantum_n_layers: 2
  quantum_rotation_config: "u3"
  quantum_entanglement: "cyclic_cnot"
  warmup_epochs: 3

triple_branch_fusion_dual_quantum:
  enabled: true
  batch_size: 8
  lr: 1e-5
  weight_decay: 1e-4
  use_amp: true
  swin_variant: "small"
  convnext_variant: "small"
  efficientnet_variant: "b5"
  dropout: 0.3
  fusion_dim: 768
  num_heads: 8
  quantum_mode: "dual_quantum"    # both CNN-FM AND post-cross-attn
  quantum_n_qubits: 8
  quantum_n_layers: 2
  quantum_rotation_config: "u3"
  quantum_entanglement: "cyclic_cnot"
  warmup_epochs: 3
```

---

## Why the User's Insight is Correct

The paper already proved:
- VQC on **rich intermediate CNN features** (CNN-FM) > VQC on **collapsed pooled features** (Quantum-Fusion)
- The richer and more structurally meaningful the VQC input, the better

The user's insight extends this naturally:
- **Cross-attention enhanced features** are richer than **single-branch spatial maps** because they contain cross-backbone information
- VQC on cross-attention-enhanced features should therefore capture correlations that are:
  1. Invisible to single-backbone VQC
  2. Invisible to classical fusion (which only uses linear combinations)
  3. Only possible through quantum superposition encoding the multi-backbone state

**In other words: the user is proposing we give the quantum circuit the most information-rich possible input, which is exactly what made CNN-FM-Quantum beat all classical variants.**

The theoretical prediction: PostCrossAttn-Quantum ≥ CNN-FM-Quantum, with DualQuantum potentially higher still (combining both information sources).

---

## Running New Variants

```bash
# Add to run_pipeline.py model registry:
python run_pipeline.py --models triple_branch_fusion_post_crossattn_quantum
python run_pipeline.py --models triple_branch_fusion_dual_quantum  
python run_pipeline.py --models triple_branch_fusion_all_branch_fm_quantum
python run_pipeline.py --models triple_branch_fusion_quantum_cross_attn  # last/risky
```
