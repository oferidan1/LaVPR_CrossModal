"""
Extract BGE from LaVPR checkpoint, merge T2T LoRA, attach fresh cross-modal LoRA.
==================================================================================
Saves a standalone HuggingFace BGE model (no LaVPR wrapper) with the T2T LoRA
merged into base weights and a fresh cross-modal LoRA adapter attached.

Two load modes:
  --lora-path   : PEFT adapter directory  (peft.PeftModel.from_pretrained)
  --model-path  : Lightning .ckpt file    (torch.load -> state_dict)

Usage:
    # from LoRA adapter directory
    python merge_and_add_lora.py \
        --bge-model   BAAI/bge-large-en-v1.5 \
        --lora-path   checkpoints/bge_large_t2t_lora \
        --output-dir  checkpoints/bge_cross_modal_init \
        --lora-r 64 --lora-target all

    # from Lightning checkpoint
    python merge_and_add_lora.py \
        --bge-model   BAAI/bge-large-en-v1.5 \
        --model-path  checkpoints/epoch=23.ckpt \
        --output-dir  checkpoints/bge_cross_modal_init \
        --lora-r 64 --lora-target all
"""

import argparse
import logging
from pathlib import Path

import torch
import peft
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

LORA_TARGETS = {
    "qv":         ["query", "value"],
    "all":        ["query", "key", "value", "dense"],
    "all-linear": "all-linear",   # PEFT keyword — targets every nn.Linear
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract BGE from LaVPR ckpt, merge T2T LoRA, attach cross-modal LoRA."
    )

    # ── Source ────────────────────────────────────────────────────────────────
    p.add_argument("--bge-model", default="BAAI/bge-large-en-v1.5",
                   help="HuggingFace model name or local path for BGE base")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--lora-path", default=None,
                     help="PEFT adapter directory saved by on_save_checkpoint")
    src.add_argument("--model-path", default=None,
                     help="Lightning .ckpt file")

    # ── Output / new LoRA ─────────────────────────────────────────────────────
    p.add_argument("--output-dir", required=True)
    p.add_argument("--lora-r",        type=int,   default=64)
    p.add_argument("--lora-alpha",    type=int,   default=None,
                   help="Defaults to 2 * lora-r")
    p.add_argument("--lora-target",   default="all-linear",
                   choices=["qv", "all", "all-linear"])
    p.add_argument("--lora-dropout",  type=float, default=0.1)
    p.add_argument("--device",        default="cpu")
    p.add_argument("--save-merged-only", action="store_true", default=True,
                   help="Just save the merged BGE base model, skip attaching new LoRA")

    args = p.parse_args()
    if args.lora_alpha is None:
        args.lora_alpha = 2 * args.lora_r
    return args


def load_peft_bge(args):
    """
    Returns a PeftModel wrapping a plain AutoModel BGE,
    with T2T LoRA weights loaded — ready for merge_and_unload().
    No LaVPR class involved.
    """
    log.info(f"Loading BGE base: {args.bge_model}")
    base = AutoModel.from_pretrained(args.bge_model, attn_implementation="sdpa")

    if args.lora_path:
        # ── saved by LaVPR.on_save_checkpoint → vlm_encoder.save_pretrained ──
        log.info(f"Loading PEFT adapter: {args.lora_path}")
        model = peft.PeftModel.from_pretrained(base, args.lora_path,
                                               is_trainable=False)
        return model, None

    else:
        # ── Lightning checkpoint: extract only vlm_encoder weights ────────────
        log.info(f"Loading Lightning checkpoint: {args.model_path}")
        full_sd = torch.load(args.model_path, map_location=args.device)['state_dict']

        # keep only keys that belong to vlm_encoder, strip the prefix
        bge_sd = {}
        for k, v in full_sd.items():
            for prefix in ("vlm_encoder.", "text_encoder."):   # handle both naming conventions
                if k.startswith(prefix):
                    bge_sd[k[len(prefix):]] = v
                    break

        log.info(f"Extracted {len(bge_sd)} keys from checkpoint for vlm_encoder")

        lora_keys = [k for k in bge_sd if "lora_" in k]
        log.info(f"  base_layer keys : {sum(1 for k in bge_sd if 'base_layer' in k)}")
        log.info(f"  lora keys       : {len(lora_keys)}")

        if not lora_keys:
            # plain merged checkpoint — no LoRA at all, load directly into AutoModel
            log.info("No LoRA keys — plain merged checkpoint, loading directly into AutoModel")
            # strip base_model.model. prefix if present
            clean_sd = {}
            for k, v in bge_sd.items():
                k2 = k.replace("base_model.model.", "").replace("base_layer.", "")
                clean_sd[k2] = v
            missing, unexpected = base.load_state_dict(clean_sd, strict=False)
            if missing:
                log.warning(f"Missing keys: {missing[:3]} ...")
            return None, base

        # ── all-linear PEFT format ─────────────────────────────────────────────
        # Keys look like:
        #   encoder.layer.0.attention.self.query.base_layer.weight  (base weights)
        #   encoder.layer.0.attention.self.query.lora_A.default.weight  (LoRA)
        # We infer the config, build a PeftModel, then load the full bge_sd directly.
        lora_config = _infer_lora_config(lora_keys, bge_sd)
        model = get_peft_model(base, lora_config)

        # build the state dict the PeftModel expects:
        # PeftModel keys: base_model.model.encoder.layer.0.attention.self.query.base_layer.weight
        peft_sd = {"base_model.model." + k: v for k, v in bge_sd.items()}
        missing, unexpected = model.load_state_dict(peft_sd, strict=False)
        if missing:
            log.warning(f"Missing keys ({len(missing)}): {missing[:3]} ...")
        if unexpected:
            log.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:3]} ...")
        log.info(f"Loaded {len(bge_sd)} keys into PeftModel")

    return model, None


def _infer_lora_config(lora_keys, full_sd):
    """Infer LoRA rank and target modules from checkpoint keys.
    
    Handles both standard PEFT format:
      encoder.layer.0.attention.self.query.lora_A.weight
    and all-linear format:
      encoder.layer.0.attention.self.query.lora_A.default.weight
    """
    target_modules = set()
    r = None
    for k in lora_keys:
        if "lora_A" in k:
            parts = k.split(".")
            idx = parts.index("lora_A")
            mod = parts[idx - 1]
            target_modules.add(mod)
            if r is None:
                v = full_sd[k]
                r = v.shape[0]  # lora_A shape: [r, in_features]

    log.info(f"Inferred LoRA rank={r}, target_modules={sorted(target_modules)}")

    # if all-linear was used, target_modules covers every linear layer —
    # pass "all-linear" to PEFT so it rebuilds the same structure
    if len(target_modules) > 6:
        log.info("Many target modules detected — using 'all-linear'")
        targets = "all-linear"
    else:
        targets = sorted(target_modules)

    return LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=r or 64,
        lora_alpha=(r or 64) * 2,
        target_modules=targets,
        bias="none",
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: load BGE weights ───────────────────────────────────────────────
    peft_model, merged = load_peft_bge(args)

    # ── Step 2: merge T2T LoRA (skip if already merged) ───────────────────────
    if peft_model is not None:
        log.info("Merging T2T LoRA into base weights...")
        merged = peft_model.merge_and_unload()
        log.info("Merge complete — result is a plain HuggingFace AutoModel")
    else:
        log.info("Checkpoint is already a plain merged model — skipping merge step")

    # ── Step 2b: save merged base only and exit ───────────────────────────────
    if args.save_merged_only:
        base_out = Path(args.output_dir) / "base_model"
        log.info(f"Saving merged BGE base → {base_out}")
        merged.save_pretrained(base_out)
        AutoTokenizer.from_pretrained(args.bge_model).save_pretrained(base_out)
        log.info("Done — plain HuggingFace BGE model saved (no LoRA attached)")
        log.info(f"Reload: AutoModel.from_pretrained('{base_out}')")
        return

    # ── Step 3: attach fresh cross-modal LoRA ─────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=LORA_TARGETS[args.lora_target],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    log.info(
        f"Attaching cross-modal LoRA: r={args.lora_r}, alpha={args.lora_alpha}, "
        f"target={args.lora_target} {LORA_TARGETS[args.lora_target]}"
    )
    model = get_peft_model(merged, lora_config)
    model.print_trainable_parameters()

    # ── Step 4: save ───────────────────────────────────────────────────────────
    base_out = output_dir / "base_model"          # plain HF AutoModel
    lora_out = output_dir / "cross_modal_lora"    # fresh uninitialised adapter

    log.info(f"Saving merged BGE base → {base_out}")
    merged.save_pretrained(base_out)              # saves as plain HF model, no PEFT

    log.info(f"Saving cross-modal LoRA adapter → {lora_out}")
    model.save_pretrained(lora_out)

    AutoTokenizer.from_pretrained(args.bge_model).save_pretrained(base_out)

    log.info("Done.")
    log.info(f"  Merged BGE base (plain HF) : {base_out}")
    log.info(f"  Cross-modal LoRA adapter   : {lora_out}")
    log.info("Reload for cross-modal training:")
    log.info(f"  base  = AutoModel.from_pretrained('{base_out}')")
    log.info(f"  model = PeftModel.from_pretrained(base, '{lora_out}')")


if __name__ == "__main__":
    main()