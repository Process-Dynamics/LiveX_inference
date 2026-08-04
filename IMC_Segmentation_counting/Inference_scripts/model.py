"""v5a model: DINOv3 ViT-B/16 + MCTformer+ class tokens + DPT decoder + heads.

Architecture (see docs/superpowers/specs/2026-05-14-beta-seg-v5-design.md):

  pixel_values
      |
      v
  DINOv3 ViT-B/16 (hacked with 6 class tokens in cls + register slots)
   - returns hidden_states (L+1 tensors) and last_hidden_state
   - extracts:
     - class_tokens_per_layer (L tensors, (B, C, D) each)  [for CCT loss]
     - patch_features at out_indices [3, 6, 9, 12]          [for DPT + fusion]
     - patch_features_final                                  [for cls_patch via 3x3 Conv]
   |
   v
  Fusion (only if use_ref=True):
   - For each i in out_indices:
       diff_i  = cur_i - ref_i               (raw signed, NO InstanceNorm)
       delta_i = SE(Conv3x3([ref_i, diff_i]))
       fused_i = cur_i + alpha_i * delta_i   (alpha_i init 0)
   - If use_ref=False:
       fused_i = cur_i
   |
   v
  DPT Neck + Head (HuggingFace transformers DPT)
   - input: fused features at 4 hooks
   - output: binary seg logits (B, 2, H, W)
   |
   v
  Heads:
   - binary seg head: DPT output (B, 2, H, W)
   - cls_class head:  class_tokens_final.mean(-1) -> (B, num_classes_total)
   - cls_patch head:  Conv3x3 on patch_features_final -> (B, num_classes_total, H_p, W_p)
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import AutoConfig, AutoModel
from transformers.models.dpt.configuration_dpt import DPTConfig
from transformers.models.dpt.modeling_dpt import (
    DPTNeck,
    DPTSemanticSegmentationHead,
)


# ============================================================================
# GWRP (Global Weighted Ranking Pool) — used by cls_patch image-level loss
# ============================================================================

def gwrp_pool(patch_cam: torch.Tensor, decay: float = 0.996) -> torch.Tensor:
    """Global Weighted Ranking Pool (MCTformer+ aligned).

    Args:
        patch_cam: (B, C, H, W).
        decay:     geometric decay d.

    Returns:
        class_scores: (B, C).
    """
    B, C, H, W = patch_cam.shape
    flat = patch_cam.view(B, C, -1)
    sorted_vals, _ = flat.sort(dim=-1, descending=True)
    K = sorted_vals.shape[-1]
    weights = decay ** torch.arange(K, device=flat.device, dtype=torch.float32)
    weights = weights / weights.sum()
    weights = weights.to(flat.dtype).view(1, 1, K)
    return (sorted_vals * weights).sum(dim=-1)


# ============================================================================
# Fusion module (Option B + Residual, used when use_ref=True)
# ============================================================================

class SqueezeExcitation(nn.Module):
    """Standard SE block."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        h = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, h)
        self.fc2 = nn.Linear(h, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        gap = x.mean(dim=(-2, -1))                     # (B, C)
        gap = F.relu(self.fc1(gap), inplace=True)
        gate = torch.sigmoid(self.fc2(gap))             # (B, C)
        return x * gate.unsqueeze(-1).unsqueeze(-1)


class FusionBlockOptionBResidual(nn.Module):
    """Fusion module with residual identity.

      diff_i  = cur_i - ref_i              # raw signed, NO IN
      delta_i = SE(Conv3x3([ref_i, diff_i]))
      fused_i = cur_i + alpha_i * delta_i  # alpha init 0
    """
    def __init__(self, dim: int, se_reduction: int = 16, alpha_init: float = 0.0):
        super().__init__()
        self.conv3x3 = nn.Conv2d(2 * dim, dim, kernel_size=3, padding=1)
        self.se = SqueezeExcitation(dim, reduction=se_reduction)
        self.alpha = nn.Parameter(torch.full((1,), float(alpha_init)))

    def forward(self, cur_i: torch.Tensor, ref_i: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        diff_i = cur_i - ref_i
        concat = torch.cat([ref_i, diff_i], dim=1)
        delta = self.se(self.conv3x3(concat))
        fused = cur_i + self.alpha * delta
        # diagnostics
        with torch.no_grad():
            alpha_val = float(self.alpha.item())
            # effective signal strength: ||alpha * delta|| / ||cur||
            delta_norm = (self.alpha * delta).norm()
            cur_norm = cur_i.norm() + 1e-8
            eff_ratio = float((delta_norm / cur_norm).item())
        return fused, {"alpha": alpha_val, "eff_ratio": eff_ratio}


# ============================================================================
# Encoder wrapper: DINOv3 + MCTformer+ class token hack
# ============================================================================

class DinoV3Encoder(nn.Module):
    """DINOv3 ViT-B/16 with 6 class tokens injected (v4_cct hack).

    Frozen tokens: indices in `frozen_class_indices` get requires_grad=False
    after construction; they retain whatever values the v4_cct ckpt held.

    Forward returns:
        class_tokens_per_layer: List[L] of (B, C, D)
        class_tokens_final:     (B, C, D)
        patch_features_at_outs: dict {i: (B, D, H_p, W_p)} for each i in out_indices
        patch_features_final:   (B, D, H_p, W_p)
    """
    def __init__(
        self,
        backbone_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        num_classes_total: int = 6,
        out_indices: List[int] = (3, 6, 9, 12),
        attn_implementation: str = "sdpa",
    ):
        super().__init__()
        self.num_classes_total = num_classes_total
        self.out_indices = list(out_indices)

        # Load DINOv3
        self.backbone = AutoModel.from_pretrained(
            backbone_name,
            attn_implementation=attn_implementation,
        )
        self.embed_dim = self.backbone.config.hidden_size

        # --- MCTformer+ hack: inject class tokens via cls + register slots ---
        with torch.no_grad():
            orig_cls = self.backbone.embeddings.cls_token.data.clone().contiguous()
            orig_reg = self.backbone.embeddings.register_tokens.data.clone().contiguous()
        self.R_orig = int(orig_reg.shape[1])

        new_cls_param = nn.Parameter(orig_cls.clone().contiguous())
        if num_classes_total - 1 > 0:
            extra_cls = orig_cls.expand(1, num_classes_total - 1, -1).contiguous()
            new_reg_init = torch.cat([extra_cls, orig_reg], dim=1).contiguous()
        else:
            new_reg_init = orig_reg.clone().contiguous()
        new_reg_param = nn.Parameter(new_reg_init)

        self.backbone.embeddings.cls_token = new_cls_param
        self.backbone.embeddings.register_tokens = new_reg_param
        self.backbone.config.num_register_tokens = num_classes_total - 1 + self.R_orig

        # Freeze any embedding params not on the forward path (e.g., mask_token)
        for name in ("mask_token",):
            if hasattr(self.backbone.embeddings, name):
                p = getattr(self.backbone.embeddings, name)
                if isinstance(p, nn.Parameter):
                    p.requires_grad_(False)

        self.num_layers = self.backbone.config.num_hidden_layers

        # Track class-token-slot grad-hook handles so we can remove old ones
        # before re-registering. PyTorch register_hook accumulates; without
        # removal we'd stack one duplicate hook per freeze/unfreeze cycle.
        self._slot_hook_handles: List = []

    # --------------------------------------------------------------------

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad_(True)
        # Reapply mask_token freeze
        if hasattr(self.backbone.embeddings, "mask_token"):
            mt = self.backbone.embeddings.mask_token
            if isinstance(mt, nn.Parameter):
                mt.requires_grad_(False)

    def _clear_slot_hooks(self):
        for h in self._slot_hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._slot_hook_handles = []

    def freeze_class_token_slots(self, frozen_indices: List[int]):
        """Freeze gradients on specific class token slots.

        Because all 6 class tokens are baked into the (cls + register_tokens) parameter
        layout, we can't directly set requires_grad on a slice. Instead, we register
        a backward hook that zeros gradients at frozen indices.

        Notes:
          - register_hook only fires on tensors with requires_grad=True (and a backward
            graph). If params are currently frozen (requires_grad=False), we skip
            registration here — caller is expected to re-call this after unfreeze.
          - We remove any previously-registered slot hooks before installing new ones
            to prevent accumulation across freeze/unfreeze cycles.
        """
        # Clear any old handles first
        self._clear_slot_hooks()

        cls_param = self.backbone.embeddings.cls_token
        reg_param = self.backbone.embeddings.register_tokens

        # Skip if params currently aren't gradient-tracked (no hook would fire anyway)
        if not (cls_param.requires_grad and reg_param.requires_grad):
            return

        def cls_hook(grad: torch.Tensor) -> torch.Tensor:
            if 0 in frozen_indices:
                return torch.zeros_like(grad)
            return grad

        def reg_hook(grad: torch.Tensor) -> torch.Tensor:
            g = grad.clone()
            for ci in frozen_indices:
                if ci == 0:
                    continue
                slot = ci - 1
                if 0 <= slot < (self.num_classes_total - 1):
                    g[:, slot, :] = 0.0
            return g

        self._slot_hook_handles.append(cls_param.register_hook(cls_hook))
        self._slot_hook_handles.append(reg_param.register_hook(reg_hook))

    # --------------------------------------------------------------------

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.backbone(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states           # tuple len = L+1
        last_hidden = outputs.last_hidden_state

        S = last_hidden.shape[1]
        prefix = self.num_classes_total + self.R_orig
        N = S - prefix
        if N <= 0:
            raise ValueError(f"Unexpected seq length {S} (prefix={prefix})")

        H_p = int(math.sqrt(N))
        W_p = N // H_p
        if H_p * W_p != N:
            raise ValueError(f"Patch grid not square: N={N}")

        # Class tokens per layer (for CCT)
        class_tokens_per_layer = [
            h[:, 0:self.num_classes_total, :] for h in hidden_states[1:]
        ]
        class_tokens_final = last_hidden[:, 0:self.num_classes_total, :]

        # Patch features at out_indices (1-indexed in our config -> use hidden_states[i])
        # hidden_states[0] = embeddings; hidden_states[i] = layer i output for i in 1..L
        def to_spatial(h: torch.Tensor) -> torch.Tensor:
            patches = h[:, prefix:, :]                          # (B, N, D)
            B = patches.shape[0]
            return (
                patches.reshape(B, H_p, W_p, self.embed_dim)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

        patch_features_at_outs = {}
        for i in self.out_indices:
            patch_features_at_outs[i] = to_spatial(hidden_states[i])

        patch_features_final = to_spatial(last_hidden)

        return {
            "class_tokens_per_layer": class_tokens_per_layer,
            "class_tokens_final":     class_tokens_final,
            "patch_features_at_outs": patch_features_at_outs,
            "patch_features_final":   patch_features_final,
            "patch_grid":             (H_p, W_p),
        }


# ============================================================================
# DPT decoder wrapper (HuggingFace DPTNeck + DPTSemanticSegmentationHead)
# ============================================================================

class DPTDecoder(nn.Module):
    """Standard HF DPT decoder for binary semantic segmentation.

    Inputs: list of 4 spatial feature maps (B, D, H_p, W_p) at out_indices.
    Output: (B, num_seg_classes, H_in, W_in) logits.
    """
    def __init__(
        self,
        backbone_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        embed_dim: int = 768,
        patch_size: int = 16,
        image_size: int = 2048,
        num_seg_classes: int = 2,
        neck_hidden_sizes: List[int] = (96, 192, 384, 768),
        fusion_hidden_size: int = 256,
        reassemble_factors: List[float] = (4, 2, 1, 0.5),
        three_head: bool = False,
    ):
        super().__init__()
        # Load backbone_config from HF Hub via AutoConfig — this follows the official
        # DPT-with-custom-backbone pattern (HF docs):
        #   https://huggingface.co/docs/transformers/model_doc/dpt
        # Setting out_features and reshape_hidden_states ensures DPTNeck gets
        # 4 backbone feature maps in (B, seq, D) format that it reshapes internally.
        # We also override image_size to match our actual input (2048, not HF default 224).
        bc = AutoConfig.from_pretrained(
            backbone_name,
            out_features=["stage1", "stage2", "stage3", "stage4"],
            reshape_hidden_states=False,
        )
        bc.image_size = image_size

        dpt_cfg = DPTConfig(
            backbone_config=bc,
            num_labels=num_seg_classes,
            neck_hidden_sizes=list(neck_hidden_sizes),
            fusion_hidden_size=fusion_hidden_size,
            reassemble_factors=list(reassemble_factors),
            readout_type="ignore",
            use_auxiliary_head=False,
            semantic_loss_ignore_index=255,
            semantic_classifier_dropout=0.1,
            patch_size=patch_size,
        )

        self.neck = DPTNeck(dpt_cfg)
        # self.head == head_total (crystal/liquid). Kept under the original attr
        # name so existing checkpoints' `decoder.head.*` keys still load (do NOT
        # rename to head_total — that would silently leave it random-init under
        # strict=False loading).
        self.head = DPTSemanticSegmentationHead(dpt_cfg)
        # 3-head Alpha/Beta separation: two extra cheap seg heads sharing self.neck.
        # head_alpha / head_beta are absent from single-head init checkpoints; on
        # fresh start head_alpha is copied from head_total (see
        # V5aModel.init_alpha_head_from_total) and head_beta is left fresh-random.
        self.three_head = bool(three_head)
        if self.three_head:
            self.head_alpha = DPTSemanticSegmentationHead(dpt_cfg)
            self.head_beta = DPTSemanticSegmentationHead(dpt_cfg)
        else:
            self.head_alpha = None
            self.head_beta = None

    def forward(
        self,
        feature_maps: List[torch.Tensor],     # list of 4 (B, D, H_p, W_p)
        out_h: int,
        out_w: int,
        patch_h: int,
        patch_w: int,
    ):
        # DPTNeck expects (B, seq_len, C) format with readout_type='ignore' stripping the first.
        # Our feature_maps are (B, D, H_p, W_p) -> flatten to (B, H_p*W_p, D), prepend dummy token.
        seq_inputs = []
        for fm in feature_maps:
            B, D, H_p, W_p = fm.shape
            assert H_p == patch_h and W_p == patch_w
            seq = fm.permute(0, 2, 3, 1).reshape(B, H_p * W_p, D)        # (B, N, D)
            dummy = torch.zeros(B, 1, D, device=seq.device, dtype=seq.dtype)
            seq_inputs.append(torch.cat([dummy, seq], dim=1))             # (B, N+1, D)

        hidden_states = self.neck(seq_inputs, patch_height=patch_h, patch_width=patch_w)

        def _head_logits(head: nn.Module) -> torch.Tensor:
            lg = head(hidden_states)
            return F.interpolate(lg, size=(out_h, out_w), mode="bilinear", align_corners=False)

        logits_total = _head_logits(self.head)
        if not self.three_head:
            return logits_total
        # neck runs ONCE; the 3 cheap heads share `hidden_states`.
        logits_alpha = _head_logits(self.head_alpha)
        logits_beta = _head_logits(self.head_beta)
        return logits_total, logits_alpha, logits_beta


# ============================================================================
# v5a model orchestrator
# ============================================================================

class V5aModel(nn.Module):
    """Full v5a model.

    Heads:
        seg:        DPT output -> binary mask logits (B, 2, H, W)
        cls_class:  class_tokens_final.mean(-1) -> (B, num_classes_total)
        cls_patch:  Conv3x3 on patch_features_final -> (B, num_classes_total, H_p, W_p)
    """

    def __init__(self, config: dict):
        super().__init__()
        m = config["model"]
        v = config["variant"]
        self.use_ref = bool(v["use_ref"])
        self.num_classes_total = int(m["num_classes_total"])
        self.active_class_indices = list(m["active_class_indices"])
        self.frozen_class_indices = list(m["frozen_class_indices"])
        self.out_indices = list(m["out_indices"])

        # ---- Encoder ----
        self.encoder = DinoV3Encoder(
            backbone_name=m["backbone_name"],
            num_classes_total=self.num_classes_total,
            out_indices=self.out_indices,
            attn_implementation="sdpa",
        )
        self.encoder.freeze_class_token_slots(self.frozen_class_indices)
        embed_dim = self.encoder.embed_dim

        # ---- Fusion (if use_ref) ----
        if self.use_ref:
            self.fusion = nn.ModuleDict({
                str(i): FusionBlockOptionBResidual(
                    dim=embed_dim,
                    se_reduction=int(m["fusion"]["se_reduction"]),
                    alpha_init=float(m["fusion"]["alpha_init"]),
                )
                for i in self.out_indices
            })
        else:
            self.fusion = None

        # ---- DPT decoder ----
        patch_size = self.encoder.backbone.config.patch_size
        self.decoder = DPTDecoder(
            backbone_name=m["backbone_name"],
            embed_dim=embed_dim,
            patch_size=patch_size,
            image_size=2048,                # matches dataset.target_size
            num_seg_classes=int(m["num_seg_classes"]),
            neck_hidden_sizes=list(m["neck_hidden_sizes"]),
            fusion_hidden_size=int(m["fusion_hidden_size"]),
            reassemble_factors=list(m["reassemble_factors"]),
            three_head=bool(m.get("three_head", False)),
        )
        self.three_head = bool(m.get("three_head", False))
        # Optional gradient checkpointing of the DPT decoder (neck + heads),
        # toggled by set_decoder_checkpointing(). OFF by default + on its own config
        # flag: in the 3-head OOM test it saved ~0 (the cost is the 3x full-res LOSS,
        # not the decoder internals), so the OOM was fixed via batch=4 + core-rotating
        # sampler instead. Kept only as an optional lever.
        self.ckpt_decoder = False

        # ---- Heads ----
        # cls_patch: 3x3 Conv (MCTformer+ aligned)
        self.cls_patch_head = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=self.num_classes_total,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    # --------------------------------------------------------------------

    def load_v4_cct_checkpoint(self, ckpt_path: str):
        """Initialize encoder weights from a v4_cct training checkpoint.

        v4_cct checkpoint structure: {'model': state_dict, 'epoch': ..., ...}
        Encoder keys under v4_cct are 'backbone.*' (mirroring DinoV3MCTformerPlus).
        In this model, encoder keys are 'encoder.backbone.*'. We strip the
        v4_cct prefix and load only the encoder portion. PatchCAM weights from
        v4_cct are NOT loaded (we have a fresh cls_patch_head whose weight
        layout matches but training objective differs).
        """
        sd_in = torch.load(ckpt_path, map_location="cpu")
        # v4_cct saves under "model_state_dict" (see DINOv3_MIL_v4/train.py:341).
        # Older checkpoints may use "model" — fall back to that, then to the raw dict.
        if isinstance(sd_in, dict):
            if "model_state_dict" in sd_in:
                sd = sd_in["model_state_dict"]
            elif "model" in sd_in:
                sd = sd_in["model"]
            else:
                sd = sd_in
        else:
            sd = sd_in

        # v4_cct keys to load into encoder:
        #   backbone.* -> encoder.backbone.*
        # We skip patch_head weights (different role in v5a).
        new_sd = {}
        for k, v_ in sd.items():
            if k.startswith("backbone."):
                new_sd["encoder." + k] = v_
        missing, unexpected = self.load_state_dict(new_sd, strict=False)
        loaded = len(new_sd)
        print(f"[load_v4_cct] loaded {loaded} backbone params, "
              f"missing={len(missing)}, unexpected={len(unexpected)}")
        return missing, unexpected

    def freeze_encoder(self):
        self.encoder.freeze_all()
        # Re-apply class token slot freezes (they need their hooks again)
        self.encoder.freeze_class_token_slots(self.frozen_class_indices)

    def unfreeze_encoder(self):
        self.encoder.unfreeze_all()
        self.encoder.freeze_class_token_slots(self.frozen_class_indices)

    def set_decoder_checkpointing(self, on: bool = True):
        """Enable/disable gradient checkpointing of the DPT decoder (neck + heads).
        Recomputes the decoder forward in backward. NOTE: gave ~0 memory benefit in
        the 3-head OOM test (the cost is the 3x full-res loss, not the decoder), so
        it's OFF by default; the OOM was fixed via batch=4 + core-rotating sampler.
        Optional lever only."""
        self.ckpt_decoder = bool(on)

    def init_alpha_head_from_total(self):
        """Copy head_total weights into head_alpha; leave head_beta fresh-random.

        ⚠️ Call this ONLY on a FRESH start and ONLY AFTER the init checkpoint has
        been loaded into self.decoder.head (head_total). Calling it before the
        load would copy random weights; calling it on resume would clobber a
        trained head_alpha. head_beta is intentionally NOT copied (it must learn
        "Alpha=background", the opposite of head_total's "Alpha=foreground"; its
        rare keep-class would be crushed by the abundant suppress-Alpha gradient
        if seeded from head_total — see THREE_HEAD_STRATEGY.md §Init).
        """
        if not getattr(self.decoder, "three_head", False):
            return
        self.decoder.head_alpha.load_state_dict(self.decoder.head.state_dict())
        print("[3head] head_alpha <- copy(head_total); head_beta left fresh-random")

    # --------------------------------------------------------------------

    def forward(
        self,
        cur: torch.Tensor,
        ref: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward.

        Args:
            cur: (B, 3, H, W) current frame.
            ref: (B, 3, H, W) reference frame (only used if self.use_ref).
        """
        H_in, W_in = cur.shape[-2:]

        # --- Encode cur (full backbone) ---
        enc_cur = self.encoder(cur)
        H_p, W_p = enc_cur["patch_grid"]

        # --- Encode ref if needed (no_grad) ---
        enc_ref = None
        if self.use_ref:
            if ref is None:
                raise ValueError("use_ref=True but ref is None")
            with torch.no_grad():
                enc_ref = self.encoder(ref)

        # --- Build fused feature maps at out_indices ---
        fusion_diag = {}
        fused_feature_maps = []
        for i in self.out_indices:
            cur_i = enc_cur["patch_features_at_outs"][i]                   # (B, D, H_p, W_p)
            if self.use_ref:
                ref_i = enc_ref["patch_features_at_outs"][i].detach()
                fused_i, diag = self.fusion[str(i)](cur_i, ref_i)
                fusion_diag[i] = diag
            else:
                fused_i = cur_i
            fused_feature_maps.append(fused_i)

        # --- DPT decoder -> binary seg logits (1 head, or 3 heads if three_head) ---
        if self.ckpt_decoder and self.training:
            # Recompute neck+heads in backward to save activations (3-head @ 2048²
            # is heavy). Pass the 4 feature maps as separate tensor args so
            # checkpoint tracks them; ints (H_in/W_in/H_p/W_p) captured via closure.
            def _run_decoder(*fmaps):
                return self.decoder(
                    list(fmaps), out_h=H_in, out_w=W_in, patch_h=H_p, patch_w=W_p,
                )
            decoder_out = torch.utils.checkpoint.checkpoint(
                _run_decoder, *fused_feature_maps, use_reentrant=False,
            )
        else:
            decoder_out = self.decoder(
                fused_feature_maps,
                out_h=H_in,
                out_w=W_in,
                patch_h=H_p,
                patch_w=W_p,
            )                                                              # (B, 2, H, W) or tuple of 3
        if isinstance(decoder_out, tuple):
            seg_logits, seg_logits_alpha, seg_logits_beta = decoder_out
        else:
            seg_logits, seg_logits_alpha, seg_logits_beta = decoder_out, None, None

        # --- Heads on cur encoder outputs (NOT on fused features for cls heads,
        #     to keep cls supervision pure to the encoder's own perception) ---
        # cls_class: mean over embed dim (MCTformer+ style)
        cls_logits = enc_cur["class_tokens_final"].mean(dim=-1)            # (B, C_total)

        # cls_patch: 3x3 Conv on patch_features_final
        patch_cam = self.cls_patch_head(enc_cur["patch_features_final"])   # (B, C_total, H_p, W_p)

        return {
            "seg_logits":               seg_logits,            # (B, 2, H, W) = head_total (back-compat)
            "seg_logits_alpha":         seg_logits_alpha,      # (B, 2, H, W) or None (3-head only)
            "seg_logits_beta":          seg_logits_beta,       # (B, 2, H, W) or None (3-head only)
            "cls_logits":               cls_logits,            # (B, C_total)
            "patch_cam":                patch_cam,             # (B, C_total, H_p, W_p)
            "class_tokens_per_layer":   enc_cur["class_tokens_per_layer"],  # List[L] (B, C_total, D)
            "class_tokens_final":       enc_cur["class_tokens_final"],     # (B, C_total, D)
            "patch_features_final":     enc_cur["patch_features_final"],   # (B, D, H_p, W_p)
            "patch_grid":               (H_p, W_p),
            "fusion_diag":              fusion_diag,           # {i: {alpha, eff_ratio}}
        }


# ============================================================================
# Parameter group builder (different LRs for encoder vs rest)
# ============================================================================

def get_param_groups(
    model: V5aModel,
    encoder_lr: float,
    decoder_lr: float,
    heads_lr: float,
    fusion_lr: float,
) -> List[Dict]:
    """Build param groups with differentiated LR.

    Groups:
        encoder:  model.encoder.*
        decoder:  model.decoder.*
        heads:    model.cls_patch_head.*
        fusion:   model.fusion.*  (only if use_ref)
    """
    enc_params, dec_params, head_params, fusion_params = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("encoder."):
            enc_params.append(p)
        elif n.startswith("decoder."):
            dec_params.append(p)
        elif n.startswith("cls_patch_head."):
            head_params.append(p)
        elif n.startswith("fusion."):
            fusion_params.append(p)
        else:
            head_params.append(p)  # fallback
    groups = []
    if enc_params:
        groups.append({"params": enc_params, "lr": encoder_lr, "name": "encoder"})
    if dec_params:
        groups.append({"params": dec_params, "lr": decoder_lr, "name": "decoder"})
    if head_params:
        groups.append({"params": head_params, "lr": heads_lr, "name": "heads"})
    if fusion_params:
        groups.append({"params": fusion_params, "lr": fusion_lr, "name": "fusion"})
    return groups
