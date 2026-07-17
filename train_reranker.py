import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch.optim import lr_scheduler, optimizer
import utils
from torch import nn

from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule, IMAGENET_MEAN_STD, BLIP_MEAN_STD, SIGLIP_MEAN_STD
import os
import argparse
from model.LaVPR_reranker import LaVPR_reranker


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    # parser.add_argument("--model_name", type=str, default="Salesforce/blip-itm-base-coco")        
    # parser.add_argument("--image_size", type=int, default="384", help="image size to vpr")
    # parser.add_argument("--embeds_dim", type=int, default=256, help="dimension of the embeddings")    
    # parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")            
    #parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch16")            
    parser.add_argument("--model_name", type=str, default="EVA02-B-16")            
    parser.add_argument("--image_size", type=int, default="224", help="image size to vpr")
    parser.add_argument("--embeds_dim", type=int, default=512, help="dimension of the embeddings")    
    # parser.add_argument("--model_name", type=str, default="llm2clip/LLM2CLIP-Openai-B-16")
    # parser.add_argument("--image_size", type=int, default="224", help="image size to vpr")
    # parser.add_argument("--embeds_dim", type=int, default=1280, help="dimension of the embeddings")
    # parser.add_argument("--model_name", type=str, default="google/siglip2-base-patch16-224")            
    # parser.add_argument("--image_size", type=int, default="224", help="image size to vpr")
    # parser.add_argument("--embeds_dim", type=int, default=768, help="dimension of the embeddings")   
    parser.add_argument("--gpu", type=str, default='0', help="gpu id(s) to use")    
    parser.add_argument("--epochs", type=int, default='8', help="number of epochs to train")    
    parser.add_argument("--train_csv", type=str, default="datasets/descriptions/gsv_cities_pos_rule_based.csv")    
    #parser.add_argument("--image_root", type=str, default="/mnt/d/data/gsv_cities/", help="root directory for images")
    parser.add_argument("--image_root", type=str, default="/home/shared/datasets/gsv_cities/", help="root directory for images")
    #parser.add_argument("--val_csv", type=str, default="datasets/descriptions/pitts30k_val_descriptions.csv")    
    parser.add_argument("--is_val", type=int, default="1", help="run validation 0=no/1=yes")
    parser.add_argument("--val_csv", type=str, default="datasets/descriptions/pitts30k_val_800_queries.csv")    
    #parser.add_argument("--val_image_root", type=str, default="/mnt/d/data/pitts30k/images/val", help="root directory for images")
    parser.add_argument("--val_image_root", type=str, default="/home/shared/datasets/pitts30k/images/val", help="root directory for images")
    parser.add_argument("--freeze_vlm", type=int, default="1", help="freeze the Vision-Language Model (VLM) backbone")
    parser.add_argument("--train_vlm", type=int, default="0", help="train vlm encoder or not. 1=lora, 2=full train")
    parser.add_argument("--batch_size", type=int, default="20", help="batch size for training")
    parser.add_argument("--val_batch_size", type=int, default="100", help="batch size for training")
    parser.add_argument("--img_per_place", type=int, default=4, help="number of images per place")    
    parser.add_argument("--pos_loss", type=int, default="0", help="multplier for positive loss, 0=no positive loss, >0 use positive loss")
    parser.add_argument("--neg_loss", type=int, default="0", help="multplier for negative loss, 0=no negative loss, >0 use negative loss")
    parser.add_argument("--opt", type=str, default="adamw", help="optimizer sgd/adam/adamw")    
    parser.add_argument("--lr", type=float, default="0.0002", help="learning rate")
    parser.add_argument("--lr_mult", type=float, default="0.5", help="learning rate")    
    parser.add_argument("--milestones", nargs="+", type=int, default=[5,7], help="milestones for lr scheduler seperated by space")
    parser.add_argument("--mined_negatives", type=int, default=16, help="num of mined negatives")    
    #parser.add_argument("--milestones", nargs="+", type=int, default=[10,16], help="milestones for lr scheduler seperated by space")    
    #parser.add_argument("--resume", type=str, default=None, help="resume training from path")        
    #parser.add_argument("--resume", type=str, default='checkpoints/clip_ms_sc_pos/epoch_18.ckpt') 
    parser.add_argument("--resume", type=str, default='LOGS/resnet50/lightning_logs/version_65_evaclip_b/checkpoints/last.ckpt')
    parser.add_argument("--mapping_path", type=str, default='datasets/gsv_cities_image_id_to_vocab_indices_v2.json', help="path to mapping from image id to vocab indices")
    
    
    args = parser.parse_args()
    
    return args            
            
if __name__ == '__main__':    
    pl.utilities.seed.seed_everything(seed=190223, workers=True)
    
    args = parse_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    dataset_mean_std = IMAGENET_MEAN_STD
    image_size = args.image_size
    if 'blip' in args.model_name.lower() or 'clip' in args.model_name.lower() or 'eva' in args.model_name.lower():
        dataset_mean_std = BLIP_MEAN_STD
    elif 'siglip' in args.model_name.lower():
        dataset_mean_std = SIGLIP_MEAN_STD
    
    val_set_names = []
    if args.is_val:
        val_set_names = ['pitts30k_val']    
        
    datamodule = GSVCitiesDataModule(
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        img_per_place=args.img_per_place,
        min_img_per_place=args.img_per_place,
        shuffle_all=False, # shuffle all images or keep shuffling in-city only
        random_sample_from_each_place=True,
        image_size=(image_size, image_size),
        num_workers=4,#28,
        show_data_stats=True,
        mean_std=dataset_mean_std,
        #val_set_names=['pitts30k_val', 'pitts30k_test', 'msls_val'], # pitts30k_val, pitts30k_test, msls_val
        val_set_names=val_set_names,
        train_image_root=args.image_root,
        train_csv=args.train_csv,
        val_image_root=args.val_image_root,
        val_csv=args.val_csv,
        mapping_json_path=args.mapping_path,        
    )

    model = LaVPR_reranker(
        #---- Encoder
        model_name=args.model_name.lower(),        
        embeds_dim=args.embeds_dim,        
        #---- Train hyperparameters        
        lr=args.lr, # 0.0002 for adam, 0.05 or sgd (needs to change according to batch size)        
        optimizer=args.opt, # sgd, adamw
        weight_decay=0.001, # 0.001 for sgd and 0 for adam,
        momentum=0.9,
        warmpup_steps=650,        
        milestones=args.milestones,
        lr_mult=args.lr_mult,
        epochs=args.epochs,        
        faiss_gpu=False,                
        freeze_vlm=args.freeze_vlm,
        train_vlm=args.train_vlm,        
        pos_loss=args.pos_loss,
        neg_loss=args.neg_loss,      
        num_mined_negatives=args.mined_negatives,      
    )
    
    if args.resume is not None:
        #model = model.load_from_checkpoint(args.resume)
        model_state_dict = torch.load(args.resume)['state_dict']
        renamed_state_dict = {}
        for old_key, value in model_state_dict.items():
            if 'text_encoder' not in old_key:
                continue
            new_key = old_key.replace('text_encoder', 'vlm_encoder')
            renamed_state_dict[new_key] = value
        model.load_state_dict(renamed_state_dict, strict=False)
        
    model = model.to('cuda')
    
    if args.is_val:    
        # model params saving using Pytorch Lightning
        # we save the best 3 models accoring to Recall@1 on pittsburg val
        checkpoint_cb = ModelCheckpoint(
            monitor='pitts30k_val/R1',
            filename=f'{"reranker"}' +
            '_epoch({epoch:02d})_step({step:04d})_R1[{pitts30k_val/R1:.4f}]_R5[{pitts30k_val/R5:.4f}]',
            auto_insert_metric_name=False,
            save_weights_only=True,
            save_top_k=3,
            mode='max',
            save_last=True)
    else:
        checkpoint_cb = ModelCheckpoint(        
            filename=f'{"reranker"}' +
            '_epoch({epoch:02d})_step({step:04d})',
            auto_insert_metric_name=False,
            save_weights_only=True,
            save_top_k=-1,
            every_n_epochs=1,
            mode='max',)

    #------------------
    # we instanciate a trainer
    trainer = pl.Trainer(
        accelerator='gpu', devices=1,
        default_root_dir=f'./LOGS/{"reranker"}', # Tensorflow can be used to viz
        num_sanity_val_steps=0, # runs a =- step before stating training
        #precision=16, # we use half precision to reduce  memory usage
        max_epochs=args.epochs,
        check_val_every_n_epoch=1, # run validation every epoch
        callbacks=[checkpoint_cb],# we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1, # we reload the dataset to shuffle the order
        log_every_n_steps=20,        
        precision="bf16"
        # fast_dev_run=True # uncomment or dev mode (only runs a one iteration train and validation, no checkpointing).
    )
    
    # # Manually call validation
    #trainer.validate(model=model, datamodule=datamodule)    
    
    print("Trainer device:", trainer.strategy.root_device)

    # we call the trainer, we give it the model and the datamodule
    trainer.fit(model=model, datamodule=datamodule)
