# Triple-Branch Cross-Attention Fusion (TBCA-Fusion) Explained

This file explains the `TripleBranch-Fusion` architecture used in this project in a step-by-step way. It is written so that a new reader can understand the purpose, data flow, model components, code structure, training path, and current implementation details without first reading the whole repository.

## 1. Where This Architecture Lives

Main implementation:

```text
src/models/fusion/triple_branch.py
```

Pipeline registration:

```text
run_pipeline.py
```

Configuration:

```text
config.yaml
```

Current fold-level outputs:

```text
outputs/TripleBranch-Fusion_binary/TripleBranch-Fusion_fold_results.csv
```

Related project context:

```text
project_docs_and_results/BINARY_CLASSIFICATION_RESULTS.md
research/quantum_placement_analysis.md
```

The model class is:

```python
class TripleBranchCrossAttention(nn.Module):
    ...
```

The factory function used by the pipeline is:

```python
def get_triple_branch_fusion(...):
    return TripleBranchCrossAttention(...)
```

## 2. One-Line Idea

TBCA-Fusion sends the same histopathology image through three different pretrained backbones, lets the branch features exchange information through cross-attention, learns how much to trust each branch, refines the fused representation, and finally predicts whether the image is benign or malignant.

The three branches are:

1. Swin Transformer: global/contextual image structure.
2. ConvNeXt: local texture and convolutional patterns.
3. EfficientNet-B3/B5 or optional Hybrid ViT branch: multi-scale CNN features or token-level ViT features.

In the current main `triple_branch_fusion` config, the third branch is EfficientNet-B5.

## 3. High-Level Data Flow

For binary classification, the model receives a batch of RGB image tensors:

```text
x: (B, 3, 224, 224)
```

Here, `B` means batch size.

The forward pass follows this order:

```text
Input image
  -> Swin branch feature extraction
  -> ConvNeXt branch feature extraction
  -> EfficientNet branch feature extraction
  -> project all branch features to 768 dimensions
  -> cross-attention feature exchange
  -> learnable weighted fusion
  -> optional quantum fusion layer
  -> self-attention refinement
  -> classifier head
  -> logits
```

The output is:

```text
logits: (B, num_classes)
```

For binary BreakHis classification:

```text
num_classes = 2
class 0 = Benign
class 1 = Malignant
```

## 4. Architecture Diagram

```text
Input image: (B, 3, 224, 224)
        |
        +--------------------+----------------------+----------------------+
        |                    |                      |                      |
        v                    v                      v                      |
  Swin-Small           ConvNeXt-Small         EfficientNet-B5              |
  global context       local texture          multi-scale CNN              |
  output: (B, 768)     output: (B, 768)       output: (B, 2048)            |
        |                    |                      |                      |
        v                    v                      v                      |
  Linear 768->768      Linear 768->768        Linear 2048->768             |
        |                    |                      |                      |
        +--------------------+----------------------+----------------------+
                             |
                             v
                 Cross-attention enhancement
                             |
        +--------------------+----------------------+----------------------+
        |                    |                      |                      |
        v                    v                      v                      |
  enhanced Swin        enhanced ConvNeXt      enhanced EfficientNet        |
  (B, 768)             (B, 768)               (B, 768)                     |
        |                    |                      |                      |
        +--------------------+----------------------+----------------------+
                             |
                             v
          Weighted fusion using softmax branch weights
                             |
                             v
                      fused: (B, 768)
                             |
                             v
                 optional quantum fusion layer
                             |
                             v
               self-attention / residual refinement
                             |
                             v
          Linear(768->256) -> GELU -> Dropout -> Linear(256->2)
                             |
                             v
                         logits: (B, 2)
```

## 5. Why Three Branches?

Histopathology images contain useful information at different visual levels.

Swin-Small helps with global context:

```text
large tissue organization, broad morphology, long-range structure
```

ConvNeXt-Small helps with local texture:

```text
cellular texture, edge-like patterns, local stain and tissue details
```

EfficientNet-B5 helps with multi-scale CNN features:

```text
compound-scaled convolutional features from small to larger spatial patterns
```

The goal is not just to average three models. The goal is to make their feature vectors interact before prediction.

## 6. Step 1: Build The Three Branches

The model creates the Swin branch first:

```python
from ..transformer import get_swin_tiny, get_swin_small, get_swin_v2_small

self.swin_branch = swin_factory[swin_variant](
    num_classes=num_classes,
    pretrained=True,
    dropout=dropout,
    freeze_backbone=freeze_backbones,
)
```

In the current config:

```yaml
swin_variant: "small"
```

The wrapper uses `timm` and removes the original classifier by using the backbone feature vector. Swin-Small outputs:

```text
swin_feat: (B, 768)
```

Then the ConvNeXt branch is created:

```python
from ..transformer import get_convnext_tiny, get_convnext_small, get_convnext_base

self.convnext_branch = convnext_factory[convnext_variant](
    num_classes=num_classes,
    pretrained=True,
    dropout=dropout,
    freeze_backbone=freeze_backbones,
)
```

In the current config:

```yaml
convnext_variant: "small"
```

ConvNeXt-Small also outputs:

```text
convnext_feat: (B, 768)
```

Finally, the third branch is normally EfficientNet:

```python
from ..efficientnet import get_efficientnet_b3, get_efficientnet_b5

if efficientnet_variant == 'b5':
    self.efficientnet_branch = get_efficientnet_b5(num_classes=num_classes)
    effnet_dim = 2048
else:
    self.efficientnet_branch = get_efficientnet_b3(num_classes=num_classes)
    effnet_dim = 1536
```

In the current config:

```yaml
efficientnet_variant: "b5"
```

EfficientNet-B5 outputs:

```text
effnet_feat: (B, 2048)
```

## 7. Step 2: Extract Features

The method responsible for this is:

```python
def extract_features(self, x):
    swin_feat = self.swin_branch.backbone(x)
    convnext_feat = self.convnext_branch.backbone(x)

    effnet_map = self.efficientnet_backbone(x)
    effnet_feat = F.adaptive_avg_pool2d(effnet_map, (1, 1)).squeeze(-1).squeeze(-1)

    return swin_feat, convnext_feat, effnet_feat
```

The important idea:

```text
The original classification heads are not used here.
Each backbone is used as a feature extractor.
```

The approximate tensor shapes in the main B5 setup are:

```text
swin_feat     = (B, 768)
convnext_feat = (B, 768)
effnet_feat   = (B, 2048)
```

EfficientNet produces a spatial feature map first, so the code applies global average pooling:

```python
effnet_feat = F.adaptive_avg_pool2d(effnet_map, (1, 1)).squeeze(-1).squeeze(-1)
```

This converts:

```text
(B, 2048, H, W) -> (B, 2048, 1, 1) -> (B, 2048)
```

## 8. Step 3: Project Every Branch To A Common Feature Size

Cross-attention and weighted addition require all branch features to have the same dimension. The model uses `fusion_dim = 768`.

```python
self.swin_proj = nn.Linear(swin_dim, fusion_dim)
self.convnext_proj = nn.Linear(convnext_dim, fusion_dim)
self.effnet_proj = nn.Linear(effnet_dim, fusion_dim)
```

For the current main model:

```text
swin_dim     = 768
convnext_dim = 768
effnet_dim   = 2048
fusion_dim   = 768
```

So the projections become:

```text
Swin:        768  -> 768
ConvNeXt:    768  -> 768
EfficientNet 2048 -> 768
```

The project method:

```python
def project_features(self, swin_feat, convnext_feat, effnet_feat):
    swin_proj = self.swin_proj(swin_feat)
    convnext_proj = self.convnext_proj(convnext_feat)
    effnet_proj = self.effnet_proj(effnet_feat)
    return swin_proj, convnext_proj, effnet_proj
```

After projection:

```text
swin_proj     = (B, 768)
convnext_proj = (B, 768)
effnet_proj   = (B, 768)
```

## 9. Step 4: Cross-Attention Between Branches

Cross-attention is the main architectural idea.

The model defines four cross-attention modules:

```python
self.swin_to_convnext = CrossAttention(dim=768, num_heads=8)
self.swin_to_effnet = CrossAttention(dim=768, num_heads=8)
self.convnext_to_swin = CrossAttention(dim=768, num_heads=8)
self.effnet_to_swin = CrossAttention(dim=768, num_heads=8)
```

The actual enhancement logic is:

```python
swin_enhanced = (
    swin_feat
    + self.swin_to_convnext(swin_feat, convnext_feat, convnext_feat)
    + self.swin_to_effnet(swin_feat, effnet_feat, effnet_feat)
)

convnext_enhanced = (
    convnext_feat
    + self.convnext_to_swin(convnext_feat, swin_feat, swin_feat)
)

effnet_enhanced = (
    effnet_feat
    + self.effnet_to_swin(effnet_feat, swin_feat, swin_feat)
)
```

Meaning:

```text
Swin receives information from ConvNeXt and EfficientNet.
ConvNeXt receives information from Swin.
EfficientNet receives information from Swin.
```

This makes Swin act like the central global-context branch. The local and multi-scale branches are refined using Swin's broader context, while Swin itself is enriched by both local texture and multi-scale CNN evidence.

## 10. How The CrossAttention Module Works

The class is:

```python
class CrossAttention(nn.Module):
    def __init__(self, dim=768, num_heads=8, dropout=0.1):
        ...
```

It creates query, key, value projections:

```python
self.q_proj = nn.Linear(dim, dim)
self.k_proj = nn.Linear(dim, dim)
self.v_proj = nn.Linear(dim, dim)
self.out_proj = nn.Linear(dim, dim)
```

For one branch pair:

```python
attended = self.swin_to_convnext(
    query=swin_feat,
    key=convnext_feat,
    value=convnext_feat,
)
```

Conceptually:

```text
query = what the Swin branch is asking for
key   = what the ConvNeXt branch can match against
value = information from ConvNeXt to pass back
```

The implementation uses a two-token trick:

```python
q_seq = torch.stack([q_proj, k_proj], dim=1)
k_seq = torch.stack([k_proj, q_proj], dim=1)
v_seq = torch.stack([v_proj, v_proj], dim=1)
```

This gives:

```text
q_seq: (B, 2, 768)
k_seq: (B, 2, 768)
v_seq: (B, 2, 768)
```

Why this matters:

```text
If each branch feature were treated as a single token, attention would produce a trivial 1x1 attention map.
By stacking query and key into a two-token sequence, the attention map becomes 2x2.
That allows the module to learn a real relationship between the two branch features.
```

Then the module reshapes into multi-head format:

```python
q_seq = q_seq.view(B, 2, num_heads, head_dim).transpose(1, 2)
k_seq = k_seq.view(B, 2, num_heads, head_dim).transpose(1, 2)
v_seq = v_seq.view(B, 2, num_heads, head_dim).transpose(1, 2)
```

With `dim = 768` and `num_heads = 8`:

```text
head_dim = 768 / 8 = 96
q_seq = (B, 8, 2, 96)
k_seq = (B, 8, 2, 96)
v_seq = (B, 8, 2, 96)
```

Attention scores:

```python
attn_scores = torch.matmul(q_seq, k_seq.transpose(-2, -1)) / sqrt(head_dim)
```

Shape:

```text
attn_scores = (B, 8, 2, 2)
```

The softmax turns scores into attention weights:

```python
attn_weights = F.softmax(attn_scores, dim=-1)
```

Then attention weights are applied to values:

```python
attn_out = torch.matmul(attn_weights, v_seq)
```

The model keeps only the output corresponding to token 0, the original query branch:

```python
attn_out = attn_out[:, :, 0, :].contiguous().view(batch_size, self.dim)
```

Finally:

```python
attn_out = self.out_proj(attn_out)
out = self.layer_norm(query + attn_out)
```

The residual connection means:

```text
The original branch feature is preserved, and the attended information is added as an enhancement.
LayerNorm stabilizes the result.
```

## 11. Step 5: Learnable Weighted Fusion

After cross-attention, the model has:

```text
swin_enhanced     = (B, 768)
convnext_enhanced = (B, 768)
effnet_enhanced   = (B, 768)
```

The model owns a learnable vector:

```python
self.branch_weights = nn.Parameter(torch.ones(3))
```

At initialization:

```text
branch_weights = [1.0, 1.0, 1.0]
```

Before use, the weights are softmax-normalized:

```python
weights = F.softmax(self.branch_weights, dim=0)
```

At initialization, softmax gives approximately:

```text
weights = [0.333, 0.333, 0.333]
```

The fused feature is:

```python
fused = (
    weights[0] * swin_feat
    + weights[1] * convnext_feat
    + weights[2] * effnet_feat
)
```

This is a global learned weighting, not a different weight per image. The same three learned weights are used for the whole model after training.

You can inspect the learned weights with:

```python
weights = model.get_branch_weights()
print(weights)
```

Example output shape:

```python
{
    "swin": 0.35,
    "convnext": 0.31,
    "efficientnet": 0.34,
}
```

These numbers would tell you how much the final trained model relies on each branch.

## 12. Step 6: Optional Quantum Fusion

The classical TBCA model does not enable this. The quantum variants do.

The relevant flag is:

```python
use_quantum_fusion: bool = False
```

If enabled:

```python
self.quantum_fusion = get_quantum_fusion_layer(
    input_dim=fusion_dim,
    hidden_dim=fusion_dim,
    n_qubits=quantum_n_qubits,
    n_layers=quantum_n_layers,
    rotation_config=quantum_rotation_config,
    entanglement=quantum_entanglement,
    dropout=dropout,
)
```

This is called after weighted fusion and before final refinement:

```python
if self.use_quantum_fusion:
    fused = self.quantum_fusion(fused)
```

The quantum fusion layer does this:

```text
(B, 768)
  -> classical compress to n_qubits
  -> tanh scaling to [-pi, pi]
  -> vectorized quantum circuit
  -> LayerNorm
  -> classical expand back to 768
  -> residual add with original fused feature
```

In the current config for the quantum fusion variant:

```yaml
use_quantum_fusion: true
use_quantum_bottleneck: false
quantum_n_qubits: 8
quantum_n_layers: 2
```

## 13. Step 7: Optional Quantum Bottleneck

The bottleneck variant enables quantum transforms earlier, after the branch features but before projection/fusion.

Flag:

```python
use_quantum_bottleneck: bool = False
```

If enabled:

```python
self.quantum_bottleneck = get_quantum_bottleneck(
    input_dim=fusion_dim,
    hidden_dim=fusion_dim,
    n_qubits=quantum_n_qubits,
    n_layers=quantum_n_layers,
    rotation_config=quantum_rotation_config,
    entanglement=quantum_entanglement,
    dropout=dropout,
    multi_branch=True,
    apply_to=['swin', 'convnext'],
)
```

Notice the current implementation applies it only to:

```text
swin
convnext
```

EfficientNet is skipped for efficiency and dimensional compatibility, because EfficientNet-B5 is 2048-dimensional before projection.

The forward call:

```python
if self.use_quantum_bottleneck:
    swin_feat, convnext_feat, effnet_feat = self.quantum_bottleneck(
        swin_feat, convnext_feat, effnet_feat
    )
```

Then normal projection and fusion continue.

## 14. Step 8: Optional Feature-Map Quantum Modes

There are two additional feature-map quantum variants:

```yaml
quantum_feature_map_mode: "cnn"
quantum_feature_map_mode: "vit"
```

### CNN Feature-Map Mode

In CNN mode, EfficientNet still acts as the third branch. The model extracts the EfficientNet feature map, pools it, and adds a quantum embedding derived from the feature map:

```python
effnet_map = self.efficientnet_backbone(x)
effnet_feat = F.adaptive_avg_pool2d(effnet_map, (1, 1)).squeeze(-1).squeeze(-1)

if self.quantum_feature_map_mode == 'cnn':
    effnet_feat = effnet_feat + self.feature_map_quantum(effnet_map)
```

This means the quantum module augments the CNN feature map before the EfficientNet branch is projected into 768 dimensions.

### ViT Feature-Map Mode

In ViT mode, the third branch becomes a small Hybrid CNN+ViT branch instead of EfficientNet:

```python
self.vit_branch = get_hybrid_vit(...)
effnet_dim = vit_d_model
self.third_branch_name = 'hybrid_vit'
```

The model uses token features:

```python
vit_tokens = self.vit_branch.extract_token_features(x)
effnet_feat = self.vit_branch.extract_cls_features(x)
effnet_feat = effnet_feat + self.feature_map_quantum(vit_tokens[:, 1:])
```

This adds a quantum-transformed summary of the non-CLS ViT tokens to the CLS representation.

## 15. Step 9: Self-Attention Refinement

After fusion, the feature is:

```text
fused = (B, 768)
```

The model applies:

```python
self.fusion_attention = SelfAttention(dim=fusion_dim, num_heads=num_heads)
```

Then:

```python
fused_refined = self.fusion_attention(fused)
```

Important implementation note:

```text
This SelfAttention module receives one fused vector per image, not a spatial/token sequence.
Because the sequence length is 1, the attention softmax itself is trivial.
However, the module still applies learned QKV projections, output projection, residual connection, and LayerNorm.
So in practice it acts like a learned residual feature refinement block.
```

The simplified shape flow:

```text
(B, 768)
  -> qkv projection
  -> reshape as one token
  -> output projection
  -> residual + LayerNorm
  -> (B, 768)
```

## 16. Step 10: Classification Head

The final classifier is:

```python
self.classifier = nn.Sequential(
    nn.Linear(fusion_dim, 256),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(256, num_classes),
)
```

With default values:

```text
fusion_dim = 768
dropout = 0.3
num_classes = 2
```

So the head is:

```text
Linear(768 -> 256)
GELU
Dropout(0.3)
Linear(256 -> 2)
```

The final logits are not probabilities yet:

```python
logits = self.classifier(fused_refined)
```

During training, `CrossEntropyLoss` expects logits directly. During evaluation, the pipeline applies softmax:

```python
probs = softmax(outputs.float())
```

## 17. Complete Forward Pass In Small Code

This is the architecture in compressed form:

```python
def forward(self, x):
    # 1. Extract branch features.
    swin_feat, convnext_feat, effnet_feat = self.extract_features(x)

    # 2. Optional quantum bottleneck before fusion.
    if self.use_quantum_bottleneck:
        swin_feat, convnext_feat, effnet_feat = self.quantum_bottleneck(
            swin_feat, convnext_feat, effnet_feat
        )

    # 3. Project every branch to 768 dimensions.
    swin_proj, convnext_proj, effnet_proj = self.project_features(
        swin_feat, convnext_feat, effnet_feat
    )

    # 4. Exchange information through cross-attention.
    swin_enhanced, convnext_enhanced, effnet_enhanced = self.apply_cross_attention(
        swin_proj, convnext_proj, effnet_proj
    )

    # 5. Fuse branches using learnable softmax weights.
    fused = self.fuse_features(
        swin_enhanced, convnext_enhanced, effnet_enhanced
    )

    # 6. Optional quantum fusion after branch fusion.
    if self.use_quantum_fusion:
        fused = self.quantum_fusion(fused)

    # 7. Refine fused representation.
    fused_refined = self.fusion_attention(fused)

    # 8. Predict class logits.
    logits = self.classifier(fused_refined)
    return logits
```

## 18. Current Config For Classical TBCA-Fusion

From `config.yaml`:

```yaml
triple_branch_fusion:
  enabled: true
  batch_size: 10
  lr: 5.0e-5
  weight_decay: 1.0e-4
  use_amp: true
  swin_variant: "small"
  convnext_variant: "small"
  efficientnet_variant: "b5"
  dropout: 0.3
  fusion_dim: 768
  num_heads: 8
  entropy_weight: 0.01
  warmup_epochs: 3
```

Meaning:

```text
batch_size = 10 because three large backbones use significant GPU memory.
lr = 5e-5 because earlier lower learning rates decayed too quickly.
warmup_epochs = 3 to stabilize multi-branch training at the start.
use_amp = true to use mixed precision for speed and lower memory use.
```

## 19. How The Pipeline Builds The Model

In `run_pipeline.py`, the pipeline checks the model name:

```python
elif model_name == 'triple_branch_fusion':
    model = get_triple_branch_fusion(
        num_classes=num_classes,
        swin_variant=model_cfg.get('swin_variant', 'small'),
        convnext_variant=model_cfg.get('convnext_variant', 'small'),
        efficientnet_variant=model_cfg.get('efficientnet_variant', 'b5'),
        dropout=model_cfg.get('dropout', 0.3),
        fusion_dim=model_cfg.get('fusion_dim', 768),
        num_heads=model_cfg.get('num_heads', 8),
        entropy_weight=model_cfg.get('entropy_weight', 0.01),
        freeze_backbones=model_cfg.get('freeze_backbones', False),
        use_quantum_fusion=model_cfg.get('use_quantum_fusion', False),
        use_quantum_bottleneck=model_cfg.get('use_quantum_bottleneck', False),
        quantum_n_qubits=model_cfg.get('quantum_n_qubits', 8),
        quantum_n_layers=model_cfg.get('quantum_n_layers', 2),
        quantum_rotation_config=model_cfg.get('quantum_rotation_config', 'ry_only'),
        quantum_entanglement=model_cfg.get('quantum_entanglement', 'cyclic'),
    )
    return model, 'TripleBranch-Fusion', 'fusion'
```

The returned display name is:

```text
TripleBranch-Fusion
```

The returned paradigm type is:

```text
fusion
```

That is why outputs are stored under:

```text
outputs/TripleBranch-Fusion_binary/
```

## 20. Training Flow

The unified training path is:

```text
run_pipeline.py
  -> load config
  -> build model
  -> create k-fold patient-level data splits
  -> train each fold
  -> evaluate on held-out test loader
  -> save fold results and plots
```

The optimizer:

```python
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
```

The scheduler:

```python
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
```

The default loss for this model:

```python
nn.CrossEntropyLoss(label_smoothing=0.1)
```

The training loop supports:

```text
mixed precision
gradient clipping
warmup epochs
early stopping
W&B logging
```

## 21. Important Implementation Note About Entropy Regularization

The model defines:

```python
def compute_weight_regularization(self):
    weights = F.softmax(self.branch_weights, dim=0)
    return entropy_regularization(
        weights.unsqueeze(0).unsqueeze(0),
        weight=self.entropy_weight,
    )
```

The intention is to discourage one branch from dominating too aggressively.

However, the current `run_pipeline.py` training path does not add this regularization term into the loss for `triple_branch_fusion`.

Current training effectively does:

```python
logits = model(images)
loss = criterion(logits, labels)
```

It does not currently do:

```python
loss = criterion(logits, labels) + model.compute_weight_regularization()
```

So `entropy_weight` is available in the model, but it is not active in the current pipeline unless the training loss is modified.

If you want to activate it, the clean place is inside `compute_loss_and_logits()` for TBCA model names.

Example:

```python
logits = model(images)
loss = criterion(logits, labels)

if hasattr(model, "compute_weight_regularization"):
    loss = loss + model.compute_weight_regularization()

return logits, loss
```

## 22. Dataset And Preprocessing Assumptions

The main dataset path comes from:

```yaml
data:
  dataset: "breakhis"
  data_dir: "data/BreaKHis_v1"
  task: "binary"
```

The dataset parser assigns labels from the path:

```text
benign    -> 0
malignant -> 1
```

Images are resized to:

```text
224 x 224
```

Training transforms include:

```text
random horizontal flip
random vertical flip
random rotation
random affine transform
color jitter
ImageNet normalization
random erasing
```

Validation/test transforms use:

```text
resize to 224 x 224
ImageNet normalization
```

This is important because the backbones are ImageNet-pretrained, so the ImageNet mean and standard deviation normalization is expected.

## 23. Cross-Validation Protocol

The project uses patient-level splitting to avoid leakage.

For k-fold training:

```text
15% held-out test set is created first.
The remaining 85% is split into 5 patient-grouped folds.
Train and validation folds do not share patients.
The held-out test set does not share patients with train/validation.
```

This matters for medical imaging. Without patient-level grouping, patches from the same patient can appear in both training and validation/test sets, which makes performance look better than it really is.

## 24. Current Classical TBCA Fold Results

From:

```text
outputs/TripleBranch-Fusion_binary/TripleBranch-Fusion_fold_results.csv
```

Fold-level results:

| Fold | Best Epoch | Val AUC | Val Best Acc | Test Acc | Test AUC | Sensitivity | Specificity | FNR | Inference ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 0.9154 | 0.8786 | 0.8119 | 0.8952 | 0.9590 | 0.6049 | 0.0410 | 61.23 |
| 2 | 19 | 0.9989 | 0.9840 | 0.8903 | 0.9210 | 0.9329 | 0.8304 | 0.0671 | 45.89 |
| 3 | 10 | 0.9071 | 0.8462 | 0.8794 | 0.8810 | 0.9354 | 0.8007 | 0.0646 | 42.11 |
| 4 | 1 | 0.9843 | 0.9452 | 0.8221 | 0.8546 | 0.8398 | 0.7972 | 0.1602 | 40.50 |
| 5 | 19 | 0.9574 | 0.9041 | 0.8671 | 0.9029 | 0.9280 | 0.7815 | 0.0720 | 44.11 |

Approximate means from those five rows:

```text
mean validation best accuracy: 0.9116
mean test accuracy:            0.8542
mean test AUC:                 0.8909
mean sensitivity:              0.9190
mean specificity:              0.7629
mean FNR:                      0.0810
```

The project-level `BINARY_CLASSIFICATION_RESULTS.md` also reports a deduplicated/high-level summary where `TripleBranch-Fusion (TBCA)` is listed as a strong fusion model with high validation accuracy and high sensitivity. If numbers differ between files, prefer the specific CSV for the exact run you are analyzing.

## 25. How To Run Only TripleBranch-Fusion

From the project root:

```bash
python run_pipeline.py --models triple_branch_fusion
```

To run the quantum fusion variant:

```bash
python run_pipeline.py --models triple_branch_fusion_quantum
```

To run the quantum bottleneck variant:

```bash
python run_pipeline.py --models triple_branch_fusion_bottleneck
```

To run the CNN feature-map quantum variant:

```bash
python run_pipeline.py --models triple_branch_fusion_cnn_featuremap_quantum
```

To run the ViT feature-map quantum variant:

```bash
python run_pipeline.py --models triple_branch_fusion_vit_featuremap_quantum
```

## 26. Common Places To Modify The Architecture

### Change branch sizes

Edit `config.yaml`:

```yaml
swin_variant: "small"
convnext_variant: "small"
efficientnet_variant: "b5"
```

Possible Swin variants from code:

```text
tiny
small
v2_small
```

Possible ConvNeXt variants from code:

```text
tiny
small
base
```

EfficientNet branch currently supports:

```text
b3
b5
```

### Change fusion dimension

Edit:

```yaml
fusion_dim: 768
```

Be careful. Cross-attention requires:

```text
fusion_dim % num_heads == 0
```

For example:

```text
768 / 8 = 96
```

That works cleanly.

### Change number of attention heads

Edit:

```yaml
num_heads: 8
```

Again, ensure:

```text
fusion_dim % num_heads == 0
```

### Freeze backbones

The constructor supports:

```python
freeze_backbones: bool = False
```

But note the config key used in the pipeline is:

```python
model_cfg.get('freeze_backbones', False)
```

So in YAML, the expected key is:

```yaml
freeze_backbones: true
```

not:

```yaml
freeze_backbone: true
```

## 27. What Makes This Different From Simple Ensemble Averaging?

A normal ensemble usually does this:

```text
model_1 logits + model_2 logits + model_3 logits -> average prediction
```

TBCA-Fusion does this:

```text
extract features from all branches
make branch features interact through cross-attention
fuse the enhanced internal representations
train one classifier on the fused representation
```

That means the model learns a shared representation before classification, instead of averaging final predictions after each model has already made up its mind.

## 28. What Makes This Different From Concatenation?

Simple concatenation would do:

```python
fused = torch.cat([swin_feat, convnext_feat, effnet_feat], dim=1)
```

That would produce:

```text
(B, 768 + 768 + 768) = (B, 2304)
```

Then a classifier would need to learn relationships between branches later.

TBCA instead does:

```text
cross-attention first
weighted fusion second
classification third
```

This forces explicit branch-to-branch feature exchange before final prediction.

## 29. Things A New Developer Should Know

1. The branch features are already pooled vectors, not spatial feature maps, by the time most cross-attention happens.
2. Cross-attention uses a two-token implementation to avoid trivial single-token attention.
3. Branch fusion weights are global learned parameters, not per-sample adaptive gates.
4. The self-attention refinement receives one fused vector, so it behaves mostly like residual learned feature refinement.
5. `entropy_weight` is passed into the model but is not currently added to the loss in `run_pipeline.py`.
6. EfficientNet-B5 makes the model much larger and slower than B3 but gives a stronger third branch.
7. The model is memory-heavy because all three pretrained backbones run in parallel.
8. Patient-level splitting is essential for trustworthy results.

## 30. Minimal Example For Debugging Shapes

Use this kind of snippet when debugging on a small batch:

```python
import torch
from src.models.fusion.triple_branch import get_triple_branch_fusion

model = get_triple_branch_fusion(
    num_classes=2,
    swin_variant="small",
    convnext_variant="small",
    efficientnet_variant="b5",
    fusion_dim=768,
    num_heads=8,
)

x = torch.randn(2, 3, 224, 224)

with torch.no_grad():
    swin_feat, convnext_feat, effnet_feat = model.extract_features(x)
    print(swin_feat.shape)      # expected: (2, 768)
    print(convnext_feat.shape)  # expected: (2, 768)
    print(effnet_feat.shape)    # expected: (2, 2048)

    swin_proj, convnext_proj, effnet_proj = model.project_features(
        swin_feat, convnext_feat, effnet_feat
    )
    print(swin_proj.shape)      # expected: (2, 768)
    print(convnext_proj.shape)  # expected: (2, 768)
    print(effnet_proj.shape)    # expected: (2, 768)

    logits = model(x)
    print(logits.shape)         # expected: (2, 2)
```

## 31. Short Summary For Presentation

TBCA-Fusion is a three-backbone fusion architecture for breast cancer histopathology classification. It combines Swin-Small for global contextual reasoning, ConvNeXt-Small for local texture modeling, and EfficientNet-B5 for multi-scale convolutional features. Each branch produces a feature vector, all vectors are projected to a shared 768-dimensional space, and cross-attention modules allow the branches to exchange information before fusion. A learnable softmax weight vector controls each branch's contribution, and the fused feature is refined before a small MLP classifier predicts benign versus malignant. The model is stronger than simple concatenation or logit averaging because it learns cross-branch relationships inside the representation space before classification.

