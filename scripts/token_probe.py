"""
Token-importance probe for LaVPR.

Which estimator actually identifies the tokens the retrieval score rests on:
corpus TF-IDF (what the SuperCLIP loss uses today), the model's own EOT
attention (the supervisor's proposal), or gradient attribution of the matching
score (the upper bound)?

Method: deletion / occlusion. Rank each query's tokens by an estimator, hide the
top-k, re-encode, and measure the degradation. The estimator whose tokens do the
most damage is the one that found the load-bearing words. No ground-truth token
labels are required anywhere, which is the point - attention is a belief, not a
label, so the test measures informativeness rather than assuming correctness.

Design note. Every earlier failure of this script came from reimplementing part
of the pipeline and having it drift from the real one. So:

  * query descriptors come from the SAME single_encoder(images, texts) call that
    eval uses, with the processor temporarily swapped for a stand-in that
    returns the mask we want. Nothing is recomputed.
  * images, texts and indices are read together from the query dataloader, so
    they cannot fall out of alignment.
  * the database/query selection mirrors eval_lavpr verbatim.

Usage (same arguments as eval_lavpr.py):

    python token_probe.py --database_folder ... --queries_folder ... \
        --queries_csv ... --image_root ... --model_path ... \
        --model_name openai/clip-vit-base-patch16 --cross_modal 2
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
try:
    from transformers import BatchEncoding
except ImportError:
    BatchEncoding = dict
from loguru import logger
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import eval_parser
from model.LaVPR_wrapper import LaVPR_wrapper
from dataloaders.test_dataset import TestDataset


# ===========================================================================
# driving the model's own text path
# ===========================================================================

def resolve_clip(module, max_depth=4):
    """The HF CLIP model, for the attention and gradient estimators. Resolved
    from the SAME module that owns the text path, not by an independent search."""
    seen, stack = set(), [(module, 0)]
    while stack:
        obj, d = stack.pop(0)
        if id(obj) in seen or d > max_depth:
            continue
        seen.add(id(obj))
        if hasattr(obj, 'text_model') and hasattr(obj, 'text_projection'):
            return obj
        for name in dir(obj):
            if name.startswith('_'):
                continue
            try:
                child = getattr(obj, name)
            except Exception:
                continue
            if isinstance(child, torch.nn.Module):
                stack.append((child, d + 1))
    raise RuntimeError("Could not locate an HF CLIP tower. Pass it explicitly, "
                       "e.g. model.single_encoder.vlm_encoder.")


class _FixedTokenization:
    """Processor stand-in: ignores its arguments and returns the ids and mask we
    hand it. Everything else forwards to the real processor, so the model cannot
    tell the difference."""

    def __init__(self, real, ids, mask):
        object.__setattr__(self, '_real', real)
        object.__setattr__(self, '_ids', ids)
        object.__setattr__(self, '_mask', mask)

    def __call__(self, *a, **kw):
        return BatchEncoding({'input_ids': self._ids,
                              'attention_mask': self._mask})

    def __getattr__(self, n):
        return getattr(object.__getattribute__(self, '_real'), n)


@contextmanager
def fixed_tokens(enc, ids, mask):
    real = enc.processor
    enc.processor = _FixedTokenization(real, ids, mask)
    try:
        yield
    finally:
        enc.processor = real


def query_descriptor(model, images, ids, mask, grad=False):
    """The query vector, produced by the same call eval uses.

    LaVPR_wrapper.encode_single unpacks different positions depending on
    self.reranker, and wraps the call in torch.no_grad(). We mirror the
    unpacking and bypass the no_grad when we need the gradient estimator.
    """
    enc = model.single_encoder
    n = ids.shape[0]
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, fixed_tokens(enc, ids, mask):
        if model.reranker:
            out = enc(images, [""] * n, return_embeddings=True)
            tf = out[2]                      # score_matrix, img, TEXT, ...
        else:
            out = enc(images, [""] * n)
            tf = out[1]                      # features, TEXT, ...
    return F.normalize(tf.float(), dim=-1)


def tokenize(processor, texts, max_len, device):
    """padding='max_length' so L is constant. Under CLIP's causal attention the
    pooled descriptor is immune to padding (padding sits after EOT), but the
    token sequence is not - and this probe operates on the token sequence."""
    try:
        enc = processor(text=list(texts), return_tensors='pt',
                        padding='max_length', truncation=True, max_length=max_len)
    except TypeError:
        enc = processor(list(texts), return_tensors='pt',
                        padding='max_length', truncation=True, max_length=max_len)
    ids = enc['input_ids'].to(device)
    mask = enc['attention_mask'].to(device) if 'attention_mask' in enc \
        else torch.ones_like(ids)
    return ids, mask


# ===========================================================================
# estimators -> (B, L) importance, higher = more important
# ===========================================================================

@torch.no_grad()
def imp_random(ids, mask, **kw):
    return torch.rand(mask.shape, device=mask.device)


@torch.no_grad()
def imp_idf(ids, mask, idf=None, **kw):
    if idf is None:
        return torch.zeros(mask.shape, device=mask.device)
    return idf[ids.clamp(0, idf.numel() - 1)].float() * mask.float()


@torch.no_grad()
def imp_position(ids, mask, **kw):
    """Earliest tokens first. THE control for the attention estimator.

    CLIP's text transformer is causal, so EOT attention is structurally biased
    toward early positions, and these descriptions run left-to-right across the
    scene. If `position` matches `attention`, the finding is about where in the
    sentence the information sits, not about attention as a saliency estimator.
    """
    L = mask.shape[1]
    order = (L - torch.arange(L, device=mask.device)).float()
    return order.unsqueeze(0).expand(mask.shape[0], L) * mask.float()


@torch.no_grad()
def eot_attention(clip, ids, mask):
    """Last-layer attention from the EOT position, head-averaged, recomputed by
    hand from q_proj / k_proj.

    Why not output_attentions=True: transformers >= 4.36 defaults CLIP to SDPA,
    whose fused kernel never materialises the attention matrix, so
    outputs.attentions comes back None instead of raising.
    """
    layer = clip.text_model.encoder.layers[-1]
    cap = {}

    def pre_hook(_m, args, kwargs):
        cap['h'] = args[0] if args else kwargs['hidden_states']

    h = layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        clip.text_model(input_ids=ids, attention_mask=mask)
    finally:
        h.remove()

    attn = layer.self_attn
    x = layer.layer_norm1(cap['h'])
    B, L, D = x.shape
    nh = getattr(attn, 'num_heads', None) or clip.text_model.config.num_attention_heads
    hd = D // nh

    q = attn.q_proj(x).view(B, L, nh, hd).transpose(1, 2)
    k = attn.k_proj(x).view(B, L, nh, hd).transpose(1, 2)

    b = torch.arange(B, device=ids.device)
    eot = ids.argmax(dim=-1)
    q_eot = q[b, :, eot]
    logits = torch.einsum('bhd,bhld->bhl', q_eot, k) * (hd ** -0.5)

    pos = torch.arange(L, device=ids.device).unsqueeze(0)
    allow = (pos <= eot.unsqueeze(1)) & mask.bool()
    logits = logits.masked_fill(~allow.unsqueeze(1), float('-inf'))
    return torch.softmax(logits.float(), dim=-1).mean(dim=1)


@torch.no_grad()
def imp_attention(ids, mask, clip=None, **kw):
    """EOT-row attention as token importance.

    The EOT's attention to itself is an artifact (pooling tokens act as
    attention sinks) and is zeroed, as is BOS. CLIP's text transformer is
    causal, so what remains still carries a positional bias toward early tokens
    - for left-to-right scene descriptions that is a bias toward one side of
    every scene. Worth stating if this estimator wins.
    """
    a = eot_attention(clip, ids, mask).clone()
    b = torch.arange(ids.shape[0], device=ids.device)
    a[b, ids.argmax(dim=-1)] = 0.0
    a[:, 0] = 0.0
    return a * mask.float()


def _grad_attrib(ids, mask, clip, model, images, score_fn):
    """Shared machinery: hook the token embeddings, backprop `score_fn(t)`,
    return |grad . embedding| per token."""
    store = {}

    def hook(_m, _i, o):
        o.requires_grad_(True)
        o.retain_grad()
        store['e'] = o
        return o

    h = clip.text_model.embeddings.token_embedding.register_forward_hook(hook)
    try:
        t = query_descriptor(model, images, ids, mask, grad=True)
        score = score_fn(t).sum()
        clip.zero_grad(set_to_none=True)
        score.backward()
        e = store['e']
        attrib = (e.grad * e).sum(-1).abs()
    finally:
        h.remove()
        clip.zero_grad(set_to_none=True)
    return (attrib * mask.float()).detach()


def imp_grad_abs(ids, mask, clip=None, model=None, images=None, gt_img=None, **kw):
    """d cos(text, its GT image) / d token.

    The naive attribution, and it under-performs on the deletion test: a token
    that raises the positive's score while raising every distractor's score
    equally contributes nothing to the RANKING, and this objective cannot tell
    those apart. Kept as the contrast to grad_margin.
    """
    return _grad_attrib(ids, mask, clip, model, images,
                        lambda t: (t * gt_img).sum(-1))


def imp_grad_margin(ids, mask, clip=None, model=None, images=None, gt_img=None,
                    gallery=None, pos_mask=None, **kw):
    """d [cos(text, GT) - cos(text, hardest non-positive)] / d token.

    R@1 is decided by the margin over the best distractor, not by the absolute
    score, so this is the quantity the retrieval decision actually rests on.
    Still a training-time signal (it uses the pair), so it belongs as a target
    for the SuperCLIP weighting rather than an inference-time weighting.
    """
    def score_fn(t):
        sims = t @ gallery.t()
        if pos_mask is not None:
            sims = sims.masked_fill(pos_mask, float('-inf'))
        return (t * gt_img).sum(-1) - sims.max(dim=-1).values

    return _grad_attrib(ids, mask, clip, model, images, score_fn)


ESTIMATORS = {
    'random': imp_random,          # floor
    'position': imp_position,      # control for attention's causal bias
    'idf': imp_idf,                # what SuperCLIP uses today
    'attention': imp_attention,    # the supervisor's proposal
    'grad_abs': imp_grad_abs,      # naive attribution
    'grad_margin': imp_grad_margin,  # attribution to the ranking margin
}


# ===========================================================================
# deletion
# ===========================================================================

def delete_topk(ids, mask, imp, k):
    """Hide the k most important tokens by zeroing the attention mask.

    Masking, not removing: every position index, the causal structure and the
    EOT location stay put, so the only variable is which tokens are visible.
    BOS and EOT are protected - removing EOT destroys the descriptor identically
    for every estimator and measures nothing.
    """
    b = torch.arange(ids.shape[0], device=ids.device)
    prot = torch.zeros_like(mask, dtype=torch.bool)
    prot[:, 0] = True
    prot[b, ids.argmax(-1)] = True

    scored = imp.masked_fill(prot | (mask == 0), float('-inf'))
    k = min(k, scored.shape[1])
    top = scored.topk(k, dim=1).indices
    out = mask.clone()
    out.scatter_(1, top, 0)
    return out


# ===========================================================================
# descriptors
# ===========================================================================

def query_loader(test_ds, args):
    """Queries in dataset order, images and texts together. shuffle=False, so
    batch order is the query order and no index bookkeeping is needed."""
    idx = list(range(test_ds.num_database,
                     test_ds.num_database + test_ds.num_queries))
    return DataLoader(Subset(test_ds, idx), batch_size=args.batch_size,
                      num_workers=args.num_workers, shuffle=False)


@torch.no_grad()
def build_gallery(model, test_ds, args, device):
    """Database descriptors, selected exactly as eval_lavpr selects them."""
    nd, D = test_ds.num_database, model.encoder_dim
    vision = np.zeros((nd, D), dtype='float32')
    text = np.zeros((nd, D), dtype='float32')

    dl = DataLoader(Subset(test_ds, list(range(nd))), batch_size=args.batch_size,
                    num_workers=args.num_workers, shuffle=False)
    for images, indices, texts in tqdm(dl, desc='gallery'):
        images = images.to(device)
        if getattr(args, 'bfloat16', False):
            images = images.bfloat16()
        idx = indices.numpy()
        d, tf = model.encode_single(images, list(texts))[:2]
        vision[idx, :] = d.cpu().float().numpy()
        text[idx, :] = tf.cpu().float().numpy()

    db = text if (args.cross_modal and args.text_only) else vision
    if not args.cross_modal:
        db = text
    return F.normalize(torch.from_numpy(db).to(device), dim=-1)


# ===========================================================================
# probe
# ===========================================================================

def run_probe(model, clip, processor, gallery, test_ds, positives, idf, args,
              ks=(1, 3, 5, 10), max_len=77, device='cuda'):
    pos_sets = [set(np.asarray(p).ravel().tolist()) for p in positives]
    res = {n: {k: {'cos': [], 'hit': []} for k in ks} for n in ESTIMATORS}
    base = {'cos': [], 'hit': []}
    skipped = 0
    q0 = 0
    checked = False

    for images, indices, texts in tqdm(query_loader(test_ds, args), desc='probe'):
        images = images.to(device)
        if getattr(args, 'bfloat16', False):
            images = images.bfloat16()
        B = len(texts)
        rows = [q0 + i for i in range(B)]
        q0 += B

        keep = [i for i, q in enumerate(rows) if pos_sets[q]]
        skipped += B - len(keep)
        if not keep:
            continue
        sel = torch.as_tensor(keep, device=device)
        texts_k = [texts[i] for i in keep]
        rows_k = [rows[i] for i in keep]
        images_k = images[sel]

        ids, mask = tokenize(processor, texts_k, max_len, device)
        gt_img = torch.stack([gallery[int(next(iter(pos_sets[q])))] for q in rows_k])

        # every positive of this query is excluded when looking for the hardest
        # distractor, otherwise the "negative" is often another correct image
        pos_mask = torch.zeros(len(rows_k), gallery.shape[0],
                               dtype=torch.bool, device=device)
        for i, q in enumerate(rows_k):
            pos_mask[i, torch.as_tensor(sorted(pos_sets[q]), device=device)] = True

        t0 = query_descriptor(model, images_k, ids, mask)
        c0 = (t0 * gt_img).sum(-1)
        p0 = (t0 @ gallery.t()).argmax(-1).cpu().numpy()
        base['cos'] += c0.cpu().tolist()
        base['hit'] += [int(p0[i] in pos_sets[q]) for i, q in enumerate(rows_k)]

        if not checked:
            checked = True
            r1 = 100.0 * float(np.mean(base['hit']))
            logger.info(f"first batch | R@1 {r1:.2f} | mean cos to GT "
                        f"{c0.mean():.4f} | {len(keep)}/{B} queries usable")
            if r1 < 1.0:
                raise RuntimeError(
                    "R@1 is ~0 on the first batch. The probe now uses the same "
                    "single_encoder call as eval, so this is upstream: the "
                    "checkpoint did not load, or gallery / get_positives() are "
                    "not in the same index space. Run eval_lavpr.py with these "
                    "exact arguments and confirm its usual R@1 first.")

        for name, fn in ESTIMATORS.items():
            if name == 'idf' and idf is None:
                continue
            imp = fn(ids, mask, clip=clip, model=model, images=images_k,
                     idf=idf, gt_img=gt_img, gallery=gallery, pos_mask=pos_mask)
            for k in ks:
                m = delete_topk(ids, mask, imp, k)
                t = query_descriptor(model, images_k, ids, m)
                res[name][k]['cos'] += (c0 - (t * gt_img).sum(-1)).cpu().tolist()
                p = (t @ gallery.t()).argmax(-1).cpu().numpy()
                res[name][k]['hit'] += [int(p[i] in pos_sets[q])
                                        for i, q in enumerate(rows_k)]

    if skipped:
        logger.warning(f"{skipped} queries had no positive and were skipped")

    out = {'none': {'r1': 100.0 * float(np.mean(base['hit'])),
                    'cos': float(np.mean(base['cos']))}}
    hits = {'none': np.asarray(base['hit'], dtype=np.int8)}
    for name in ESTIMATORS:
        if not res[name][ks[0]]['hit']:
            continue
        out[name] = {k: {'cos_drop': float(np.mean(res[name][k]['cos'])),
                         'r1': 100.0 * float(np.mean(res[name][k]['hit']))}
                     for k in ks}
        hits[name] = {k: np.asarray(res[name][k]['hit'], dtype=np.int8) for k in ks}
    return out, hits


def mcnemar(a, b):
    """Paired test on two 0/1 hit vectors over the SAME queries.

    These comparisons are paired, so the variance that matters is over the
    queries where the two estimators disagree, not over all queries. Comparing
    two proportions as if independent would need effects several times larger
    to reach significance.

    Returns (n_a_only, n_b_only, p) for the two-sided exact binomial test.
    """
    a, b = np.asarray(a), np.asarray(b)
    n01 = int(((a == 1) & (b == 0)).sum())
    n10 = int(((a == 0) & (b == 1)).sum())
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    try:
        from scipy.stats import binomtest
        p = binomtest(n01, n, 0.5).pvalue
    except ImportError:
        z = (abs(n01 - n10) - 1) / np.sqrt(n)
        from math import erfc
        p = erfc(z / np.sqrt(2))
    return n01, n10, float(p)


def report_significance(hits, ks, ref='random'):
    """Each estimator against the random control, paired, per k."""
    if ref not in hits:
        return
    logger.info("")
    logger.info(f"paired comparison against '{ref}' (McNemar). a stronger "
                f"estimator DESTROYS more, i.e. has FEWER hits after deletion.")
    logger.info(f"{'estimator':<13}" + "".join(f"   k={k:<2}  p     " for k in ks))
    for name in ESTIMATORS:
        if name not in hits or name == ref:
            continue
        row = f"{name:<13}"
        for k in ks:
            _, _, p = mcnemar(hits[name][k], hits[ref][k])
            star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '  '
            row += f"   {p:7.4f}{star}"
        logger.info(row)


def report(out, ks):
    logger.info(f"baseline R@1 {out['none']['r1']:.2f}  "
                f"mean cos to GT {out['none']['cos']:.3f}")
    logger.info(f"{'estimator':<12}" + "".join(f"   k={k:<2} dR@1   dcos" for k in ks))
    for name in ESTIMATORS:
        if name not in out:
            continue
        row = f"{name:<12}"
        for k in ks:
            d = out['none']['r1'] - out[name][k]['r1']
            row += f"   {d:7.2f} {out[name][k]['cos_drop']:6.3f}"
        logger.info(row)
    logger.info("")
    logger.info("Larger drop = the estimator found the tokens the score rests on.")
    logger.info("'random' is the floor. 'grad' uses the pair and is the ceiling.")
    logger.info("attention > idf  -> the attention prior is worth building.")
    logger.info("attention <= idf -> TF-IDF is the better prior here; say so with this table.")
    logger.info("grad >> both     -> neither proxy is good; distil the gradient target instead.")


# ===========================================================================
# main
# ===========================================================================

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.remove()
    start = datetime.now()
    log_dir = Path("logs") / args.log_dir / ("probe_" + start.strftime("%Y-%m-%d_%H-%M-%S"))
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(sys.stdout, colorize=True,
               format="<green>{time:%H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "probe.log", level="DEBUG")
    logger.info(" ".join(sys.argv))

    device = args.device

    IMAGENET = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    BLIP = {'mean': [0.48145466, 0.4578275, 0.40821073],
            'std': [0.26862954, 0.26130258, 0.27577711]}
    SIGLIP = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}
    name = args.model_name.lower()
    mean_std = BLIP if ('clip' in name or 'blip' in name or 'eva' in name) else IMAGENET
    if 'siglip' in name:
        mean_std = SIGLIP

    if args.cross_modal <= 1:
        raise SystemExit(
            "cross_modal <= 1: there is no single_encoder and the query "
            "descriptor comes from encode_image, so the run is image->image "
            "and a token-importance probe has nothing to measure. Use the "
            "cross_modal setting your text->image numbers came from.")

    model = LaVPR_wrapper(args)

    if 'msls_challenge' in args.image_root:
        raise SystemExit("msls_challenge has no public GT; use msls_val or pitts30k.")
    test_ds = TestDataset(
        args.database_folder, args.queries_folder, args.queries_csv, args.image_root,
        mean_std=mean_std, positive_dist_threshold=args.positive_dist_threshold,
        image_size=args.image_size, use_labels=args.use_labels)
    logger.info(f"Probing on {test_ds}")

    enc = model.single_encoder
    clip = resolve_clip(enc)
    processor = enc.processor
    logger.info(f"encoder {type(enc).__name__} | clip tower {type(clip).__name__} | "
                f"reranker={model.reranker} | cross_modal={args.cross_modal} | "
                f"text_only={getattr(args, 'text_only', None)}")

    # gradient attribution needs fp32; the probe is cheap enough not to chase
    # bf16 numerics
    clip.float()
    for p in clip.parameters():
        p.requires_grad_(False)

    idf = None
    idf_path = getattr(args, 'idf_path', None) or 'datasets/gsv_cities_clip_b32_idf.pt'
    if idf_path and os.path.exists(idf_path):
        raw = torch.load(idf_path, map_location='cpu', weights_only=True)
        idf = torch.log1p(raw.clamp(min=0.0))
        idf = (idf / idf.mean().clamp(min=1e-6)).to(device)
        logger.info(f"loaded IDF from {idf_path}  ({idf.numel()} entries)")
    else:
        logger.warning(f"{idf_path} not found, skipping the idf estimator")

    gallery = build_gallery(model, test_ds, args, device)
    zero = int((gallery.norm(dim=-1) < 1e-6).sum())
    logger.info(f"gallery {tuple(gallery.shape)} | {zero} zero rows")
    if zero:
        raise RuntimeError(f"{zero} database rows were never filled.")

    positives = test_ds.get_positives()
    if len(positives) != test_ds.num_queries:
        raise RuntimeError(f"{len(positives)} positive lists but "
                           f"{test_ds.num_queries} queries - these must align.")

    ks = [1, 3, 5, 10]
    out, hits = run_probe(model, clip, processor, gallery, test_ds, positives,
                          idf, args, ks=ks, max_len=77, device=device)
    report(out, ks)
    report_significance(hits, ks)

    np.save(log_dir / "probe.npy", out, allow_pickle=True)
    flat = {'none': hits['none']}
    for name, d in hits.items():
        if name == 'none':
            continue
        for k, v in d.items():
            flat[f"{name}_k{k}"] = v
    np.savez(log_dir / "per_query_hits.npz", **flat)
    logger.info(f"saved {log_dir / 'probe.npy'} and per_query_hits.npz")


if __name__ == "__main__":
    main(eval_parser.parse_arguments())