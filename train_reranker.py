"""
train_reranker.py
=================
Stage-2 reranker training.  Mirrors train.py structure exactly.

Validation uses the real Pitts30k val set via GSVCitiesDataModule,
reporting R@k before reranking (stage-1 only) and after (stage-1 + reranker).

Usage
-----
python train_reranker.py \
    --backbone_ckpt  checkpoints/lavpr_cross_best.ckpt \
    --triplets_json  cache/gsv_triplets.json \
    --embeddings_npz cache/gsv_embeddings.npz \
    --image_root     /home/shared/datasets/gsv_cities/ \
    --train_csv      datasets/descriptions/gsv_cities_places.csv \
    --val_image_root /home/shared/datasets/pitts30k/images/val \
    --val_csv        datasets/descriptions/pitts30k_val_800_queries.csv \
    --out_dim        768 \
    --batch_size     8  \
    --gpu            0
"""

import os
import argparse
import subprocess

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule, IMAGENET_MEAN_STD

from dataloaders.rerank_dataloader import RerankDataModule
from model.reranker_module         import CrossModalRerankModule


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    a = p.add_argument

    # Backbone (frozen)
    a('--backbone_ckpt',    type=str,
      default='checkpoints/dinov2-bge-ft-lora-ms/checkpoints/'
              'lavpr_epoch(19)_step(15640)_R1[0.5663]_R5[0.8237].ckpt')

    # Pre-mined data
    a('--triplets_json',    type=str, default='cache/gsv_triplets.json')

    a('--token_path',       type=str, default='cache/local_tokens',
      help='Path to local_tokens.h5 or local_tokens/ directory of .pt files')

    # GSV train dataset (same as train.py)
    a('--image_root',       type=str,
      default='/home/shared/datasets/gsv_cities/')
    a('--train_csv',        type=str,
      default='datasets/descriptions/gsv_cities_places.csv')
    a('--mapping_path',     type=str,
      default='datasets/gsv_cities_image_id_to_vocab_indices_v3.json')
    a('--image_size',       type=int, default=224)
    a('--img_per_place',    type=int, default=4)

    # Pitts30k validation (same as train.py)
    a('--val_image_root',   type=str,
      default='/home/shared/datasets/pitts30k/images/val')
    a('--val_csv',          type=str,
      default='datasets/descriptions/pitts30k_val_800_queries.csv')
    a('--val_set_names',    type=str, default=['pitts30k_val'], nargs='+')

    # Shortlist
    a('--top_k_train',      type=int, default=100,
      help='Mined shortlist size (in gsv_triplets.json)')
    a('--max_cands',        type=int, default=50,
      help='Candidates actually loaded per step — tune to fit VRAM')
    a('--rerank_k',         type=int, default=20,
      help='Shortlist size for reranker at val / inference')
    a('--max_pos',          type=int, default=4)

    # Reranker architecture
    a('--d_model',          type=int,   default=128)
    a('--n_heads',          type=int,   default=4)
    a('--n_layers',         type=int,   default=2)
    a('--dropout',          type=float, default=0.1)
    a('--out_dim',          type=int,   default=768,
      help='Must match backbone embeds_dim')



    # Loss
    a('--margin',           type=float, default=0.3,
      help='Margin for ranking loss: pos must score margin above all negatives')

    # Optimiser
    a('--lr',               type=float, default=1e-4)
    a('--weight_decay',     type=float, default=1e-3)
    a('--warmup_steps',     type=int,   default=200)
    a('--epochs',           type=int,   default=10)
    a('--batch_size',       type=int,   default=8,
      help='Queries per batch — keep small to fit VRAM')
    a('--grad_accum',       type=int,   default=1,
      help='Gradient accumulation steps — effective batch = batch_size × grad_accum')

    # Hardware
    a('--gpu',              type=str,   default='0')
    a('--num_workers',      type=int,   default=4)
    a('--precision',        type=str,   default='bf16',
      choices=['32', '16', 'bf16'])
    a('--faiss_gpu',        action='store_true')

    # Re-mining
    a('--remine_every',     type=int,   default=0,
      help='Re-run build_rerank_dataset.py every N epochs (0=disabled)')

    # Resume
    a('--resume',           type=str,   default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Re-mining callback
# ---------------------------------------------------------------------------

class ReMiningCallback(pl.Callback):
    def __init__(self, remine_every: int, build_script_args: list):
        self.remine_every      = remine_every
        self.build_script_args = build_script_args

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch > 0 and epoch % self.remine_every == 0:
            print(f"\n[ReMiningCallback] Re-mining at epoch {epoch}...")
            subprocess.run(
                ['python', 'build_rerank_dataset.py'] + self.build_script_args,
                check=True)
            trainer.datamodule.reload_triplets()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    pl.utilities.seed.seed_everything(seed=190223, workers=True)
    torch.set_float32_matmul_precision('high')

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    print(f"\nPeak memory per step: {args.max_cands} candidates × 1 query "
          f"(loop over {args.batch_size} queries, K={args.max_cands} tokens each)")

    # ------------------------------------------------------------------
    # Train DataModule — loads precomputed tokens, backbone never runs
    # ------------------------------------------------------------------
    train_datamodule = RerankDataModule(
        triplets_json  = args.triplets_json,
        token_path     = args.token_path,
        train_csv      = args.train_csv,
        top_k_train    = args.top_k_train,
        max_cands      = args.max_cands,
        max_pos        = args.max_pos,
        batch_size     = args.batch_size,
        num_workers    = args.num_workers,
    )

    # ------------------------------------------------------------------
    # Val DataModule  (Pitts30k — same as train.py)
    # Attached to the model so validation_epoch_end can reach it via
    # self.trainer.datamodule
    # ------------------------------------------------------------------
    val_datamodule = GSVCitiesDataModule(
        batch_size                    = args.batch_size,
        img_per_place                 = args.img_per_place,
        min_img_per_place             = args.img_per_place,
        shuffle_all                   = False,
        random_sample_from_each_place = True,
        image_size                    = (args.image_size, args.image_size),
        num_workers                   = args.num_workers,
        show_data_stats               = False,
        mean_std                      = IMAGENET_MEAN_STD,
        val_set_names                 = args.val_set_names,
        train_image_root              = args.image_root,
        train_csv                     = args.train_csv,
        val_image_root                = args.val_image_root,
        val_csv                       = args.val_csv,
        mapping_json_path             = args.mapping_path,
    )

    # Attach val_datamodule onto train_datamodule so the model's
    # validation_epoch_end can reach val_datasets / val_set_names
    # via self.trainer.datamodule  (mirrors how train.py works)
    train_datamodule.val_datamodule = val_datamodule

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    if args.resume is not None:
        print(f"Resuming from {args.resume}")
        model = CrossModalRerankModule.load_from_checkpoint(
            args.resume, map_location='cpu')
    else:
        model = CrossModalRerankModule(
            backbone_ckpt   = args.backbone_ckpt,
            d_model         = args.d_model,
            n_heads         = args.n_heads,
            n_layers        = args.n_layers,
            dropout         = args.dropout,
            out_dim         = args.out_dim,
            rerank_k        = args.rerank_k,
            faiss_gpu       = args.faiss_gpu,
            margin          = args.margin,
            lr              = args.lr,
            weight_decay    = args.weight_decay,
            warmup_steps    = args.warmup_steps,
            epochs          = args.epochs,
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    checkpoint_cb = ModelCheckpoint(
        monitor                 = 'pitts30k_val/R1_reranked',
        filename                = 'reranker_'
                                  'ep({epoch:02d})_'
                                  'R1s1[{pitts30k_val/R1_stage1:.4f}]_'
                                  'R1rr[{pitts30k_val/R1_reranked:.4f}]_'
                                  'R5rr[{pitts30k_val/R5_reranked:.4f}]_'
                                  'R10rr[{pitts30k_val/R10_reranked:.4f}]_'
                                  'gain[{pitts30k_val/R1_gain:.4f}]',
        auto_insert_metric_name = False,
        save_weights_only       = True,
        save_top_k              = 3,
        mode                    = 'max',
        save_last               = True,
    )

    callbacks = [checkpoint_cb]

    if args.remine_every > 0:
        build_args = [
            '--backbone_ckpt', args.backbone_ckpt,
            '--train_csv',     args.train_csv,
            '--image_root',    args.image_root,
            '--top_k_train',   str(args.top_k_train),
            '--gpu',           args.gpu,
        ]
        callbacks.append(ReMiningCallback(
            remine_every      = args.remine_every,
            build_script_args = build_args,
        ))

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = pl.Trainer(
        accelerator                       = 'gpu',
        devices                           = [0],
        default_root_dir                  = './LOGS/reranker',
        num_sanity_val_steps              = 0,
        max_epochs                        = args.epochs,
        check_val_every_n_epoch           = 1,
        callbacks                         = callbacks,
        reload_dataloaders_every_n_epochs = 1,
        accumulate_grad_batches           = args.grad_accum,
        log_every_n_steps                 = 20,
        gradient_clip_val                 = 1.0,
        gradient_clip_algorithm           = 'norm',
        precision                         = args.precision,
    )

    trainer.fit(model=model, datamodule=train_datamodule)