import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule, IMAGENET_MEAN_STD
import os
import argparse
from model.LaVPR_text import LaVPR_text


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    a = parser.add_argument

    # Model
    a("--model_name",    type=str,   default="BAAI/bge-large-en-v1.5",                              help="text encoder model name")
    a("--embeds_dim",    type=int,   default=1024,                                                   help="adapter output dim (BGE-large native dim)")
    a("--pooling",       type=str,   default="gem",                                                   help="pooling type: gem / mean / cls")
    a("--gem_p",         type=float, default=3.0,                                                    help="GeM pooling power (used when pooling=gem)")
    # Hardware
    a("--gpu",           type=str,   default='0',                                                    help="GPU id(s) to use")
    # Training schedule
    a("--epochs",        type=int,   default=20,                                                     help="number of training epochs")
    a("--batch_size",    type=int,   default=120,                                                    help="number of places per batch (texts = batch_size * img_per_place)")
    a("--img_per_place", type=int,   default=4,                                                      help="descriptions per place")
    a("--opt",           type=str,   default="adamw",                                                help="optimizer: sgd / adam / adamw")
    a("--lr",            type=float, default=2e-4,                                                   help="learning rate")
    a("--lr_mult",       type=float, default=0.5,                                                    help="LR decay factor for MultiStepLR")
    a("--milestones",    type=int,   default=[10, 16],          nargs="+",                           help="epoch milestones for LR scheduler")
    # Dataset
    a("--train_csv",     type=str,   default="datasets/descriptions/gsv_cities_descriptions.csv",   help="training CSV")
    a("--image_root",    type=str,   default="/home/shared/datasets/gsv_cities/",                    help="training image root")
    a("--mapping_path",  type=str,   default='datasets/gsv_cities_image_id_to_vocab_indices_v3.json', help="path to mapping JSON")
    # Validation
    a("--is_val",        type=int,   default=1,                                                      help="run validation: 0=no / 1=yes")
    a("--val_csv",       type=str,   default="datasets/descriptions/pitts30k_val_800_queries.csv",   help="validation CSV")
    a("--val_image_root",type=str,   default="/home/shared/datasets/pitts30k/images/val",            help="validation image root")
    # Loss
    a("--loss_name",     type=str,   default="MultiSimilarityLossCM",                                help="loss function name")
    # Misc
    a("--resume",        type=str,   default=None,                                                   help="checkpoint path to resume from")

    return parser.parse_args()


if __name__ == '__main__':
    pl.utilities.seed.seed_everything(seed=190223, workers=True)
    torch.set_float32_matmul_precision("high")

    args = parse_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    val_set_names = ['pitts30k_val'] if args.is_val else []

    datamodule = GSVCitiesDataModule(
        batch_size=args.batch_size,
        img_per_place=args.img_per_place,
        min_img_per_place=args.img_per_place,
        shuffle_all=False,
        random_sample_from_each_place=True,
        image_size=(224, 224),
        num_workers=4,
        show_data_stats=True,
        mean_std=IMAGENET_MEAN_STD,
        val_set_names=val_set_names,
        train_image_root=args.image_root,
        train_csv=args.train_csv,
        val_image_root=args.val_image_root,
        val_csv=args.val_csv,
        mapping_json_path=args.mapping_path,
    )

    model = LaVPR_text(
        text_model_name=args.model_name,
        embeds_dim=args.embeds_dim,
        lr=args.lr,
        optimizer=args.opt,
        weight_decay=0.001,
        momentum=0.9,
        warmpup_steps=650,
        milestones=args.milestones,
        lr_mult=args.lr_mult,
        epochs=args.epochs,
        loss_name=args.loss_name,
        miner_name='MultiSimilarityMiner',
        miner_margin=0.1,
        faiss_gpu=False,
        pooling=args.pooling,
        gem_p=args.gem_p,
    )

    if args.resume is not None:
        model = LaVPR_text.load_from_checkpoint(args.resume)

    model = model.to('cuda')

    if args.is_val:
        checkpoint_cb = ModelCheckpoint(
            monitor='pitts30k_val/R1',
            filename='lavpr_text_epoch({epoch:02d})_step({step:04d})_R1[{pitts30k_val/R1:.4f}]',
            auto_insert_metric_name=False,
            save_weights_only=True,
            save_top_k=3,
            mode='max',
            save_last=True,
        )
    else:
        checkpoint_cb = ModelCheckpoint(
            filename='lavpr_text_epoch({epoch:02d})_step({step:04d})',
            auto_insert_metric_name=False,
            save_weights_only=True,
            save_top_k=-1,
            every_n_epochs=1,
        )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],
        default_root_dir='./LOGS/lavpr_text',
        num_sanity_val_steps=0,
        max_epochs=args.epochs,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_cb],
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=20,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        precision="bf16",
    )

    trainer.fit(model=model, datamodule=datamodule)