import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import utils
from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule, IMAGENET_MEAN_STD
import os
import argparse
from model.LaVPR_cross import LaVPR_cross


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    a = parser.add_argument

    # Model
    a("--text_model_name",     type=str,   default="checkpoints/bge-l-lora-ms-merged",                help="text encoder model name")
    a("--image_model_name",    type=str,   default="dinov2",                                               help="image encoder: 'dinov2' or 'sela'")
    a("--image_size",          type=int,   default=224,                                                    help="input image size")
    a("--embeds_dim",          type=int,   default=768,                                                    help="shared projection dim for both encoders")
    # Hardware
    a("--gpu",                 type=str,   default='0',                                                    help="GPU id(s) to use")
    # Training schedule
    a("--epochs",              type=int,   default=20,                                                     help="number of training epochs")
    a("--batch_size",          type=int,   default=20,                                                     help="batch size")
    a("--img_per_place",       type=int,   default=4,                                                      help="images per place")
    a("--opt",                 type=str,   default="adamw",                                                help="optimizer: sgd / adam / adamw")
    a("--lr",                  type=float, default=2e-5,                                                   help="learning rate")
    a("--lr_mult",             type=float, default=0.5,                                                    help="LR decay factor for MultiStepLR")
    a("--milestones",          type=int,   default=[10, 16],          nargs="+",                           help="epoch milestones for LR scheduler")
    # Dataset
    a("--train_csv",           type=str,   default="datasets/descriptions/gsv_cities_descriptions.csv",   help="training CSV")
    a("--image_root",          type=str,   default="/home/shared/datasets/gsv_cities/",                    help="training image root")
    a("--mapping_path",        type=str,   default='datasets/gsv_cities_image_id_to_vocab_indices_v3.json', help="path to mapping JSON")
    # Validation
    a("--is_val",              type=int,   default=1,                                                      help="run validation: 0=no / 1=yes")
    a("--val_csv",             type=str,   default="datasets/descriptions/pitts30k_val_800_queries.csv",   help="validation CSV")
    a("--val_image_root",      type=str,   default="/home/shared/datasets/pitts30k/images/val",            help="validation image root")
    # Encoder freezing / LoRA
    a("--freeze_text",         type=int,   default=1,                                                      help="freeze text encoder: 0=no / 1=yes")
    a("--train_text",          type=int,   default=0,                                                      help="text encoder training: 0=frozen, 1=LoRA, 2=full")
    a("--freeze_image",        type=int,   default=1,                                                      help="freeze image encoder: 0=no / 1=yes")
    a("--train_image",         type=int,   default=0,                                                      help="image encoder training: 0=frozen, 1=LoRA, 2=full")
    a("--lora_all_linear",     type=int,   default=1,                                                      help="LoRA on all linear layers: 0=no / 1=yes")
    a("--lora_target_modules", type=str,   default=["query", "value"], nargs='+',                          help="LoRA target modules (when lora_all_linear=0)")
    a("--lora_r",              type=int,   default=64,                                                     help="LoRA rank")
    # Loss
    a("--loss_name",           type=str,   default="MultiSimilarityLossCM",                               help="loss function name")
    a("--unimodal_loss",       type=float, default=0.0,                                                    help="auxiliary unimodal text loss weight (0=disabled)")
    a("--pos_loss",            type=int,   default=0,                                                      help="positive-flip loss: 0=no / 1=yes")
    # Misc
    a("--resume",              type=str,   default=None,                                                   help="checkpoint path to resume from")
    a("--is_llp",              type=int,   default=0,                                                      help="LLP (CLS-reweighting) pooling: 0=no / 1=yes")

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
        image_size=(args.image_size, args.image_size),
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

    model = LaVPR_cross(
        text_model_name=args.text_model_name,
        image_model_name=args.image_model_name,
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
        freeze_text=args.freeze_text,
        train_text=args.train_text,
        freeze_image=args.freeze_image,
        train_image=args.train_image,
        lora_all_linear=args.lora_all_linear,
        lora_target_modules=args.lora_target_modules,
        lora_r=args.lora_r,
        unimodal_loss=args.unimodal_loss,
        pos_loss=args.pos_loss,
        is_llp=args.is_llp,
    )

    if args.resume is not None:
        model = LaVPR_cross.load_from_checkpoint(args.resume)

    model = model.to('cuda')

    if args.is_val:
        checkpoint_cb = ModelCheckpoint(
            monitor='pitts30k_val/R1',
            filename='lavpr_epoch({epoch:02d})_step({step:04d})_R1[{pitts30k_val/R1:.4f}]_R5[{pitts30k_val/R5:.4f}]',
            auto_insert_metric_name=False,
            save_weights_only=True,
            save_top_k=3,
            mode='max',
            save_last=True,
        )
    else:
        checkpoint_cb = ModelCheckpoint(
            filename='lavpr_epoch({epoch:02d})_step({step:04d})',
            auto_insert_metric_name=False,
            save_weights_only=True,
            save_top_k=-1,
            every_n_epochs=1,
        )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],
        default_root_dir='./LOGS/lavpr',
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