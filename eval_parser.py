import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--positive_dist_threshold", type=int, default=25, help="distance (in meters) for a prediction to be considered a positive")        
    
    # parser.add_argument("--database_folder", type=str, default="/mnt/d/data/amstertime/test/database")    
    # parser.add_argument("--queries_folder", type=str, default="/mnt/d/data/amstertime/test/queries")        
    # parser.add_argument("--image_root", type=str, default="/mnt/d/data/amstertime/test")
    
    parser.add_argument("--database_folder", type=str, default="/home/shared/datasets/amstertime/test/database")    
    parser.add_argument("--queries_folder", type=str, default="/home/shared/datasets/amstertime/test/queries")        
    parser.add_argument("--image_root", type=str, default="/home/shared/datasets/amstertime/test")
    parser.add_argument("--queries_csv", type=str, default="datasets/descriptions/amstertime_descriptions.csv")
    # parser.add_argument("--queries_csv", type=str, default="datasets/descriptions/amstertime_fixed_1.csv")
    #parser.add_argument("--queries_csv", type=str, default="datasets/descriptions/amstertime_compressed_bad.csv")        
    #parser.add_argument("--queries_csv", type=str, default="datasets/descriptions/amstertime_descriptions_subset.csv")                

    # parser.add_argument("--database_folder", type=str, default="/home/shared/datasets/pitts30k/images/test/database")    
    # parser.add_argument("--queries_folder", type=str, default="/home/shared/datasets/pitts30k/images/test/queries")        
    # parser.add_argument("--image_root", type=str, default="/home/shared/datasets/pitts30k/images/test")
    # parser.add_argument("--queries_csv", type=str, default="datasets/descriptions/pitts30k_test_descriptions.csv")    
    # parser.add_argument("--queries_csv", type=str, default="datasets/descriptions/pitts30k_test_fixed_1.csv")    

    # parser.add_argument("--database_folder", type=str, default="/home/shared/datasets/pitts30k/images/val/database")    
    # parser.add_argument("--queries_folder", type=str, default="/home/shared/datasets/pitts30k/images/val/queries")    
    # parser.add_argument("--image_root", type=str, default="/home/shared/datasets/pitts30k/images/val")    
    # parser.add_argument("--queries_csv", type=str, default="datasets/descriptions/pitts30k_val_800_queries.csv")    
       
    # parser.add_argument("--database_folder", type=str, default="/home/shared/datasets/msls/val/database")    
    # parser.add_argument("--queries_folder", type=str, default="/home/shared/datasets/msls/val/query")   
    # parser.add_argument("--image_root", type=str, default="/home/shared/datasets/msls/val/")    
    # # parser.add_argument("--queries_csv", type=str, default="datasets/descriptions/msls_val_descriptions.csv")
    # # parser.add_argument("--queries_csv", type=str, default="datasets/descriptions/msls_val_fixed_1.csv")    
    # # # # #parser.add_argument("--queries_csv", type=str, default="datasets/descriptions/msls_val_descriptions_blur.csv")
    # parser.add_argument("--queries_csv", type=str, default="datasets/descriptions/msls_val_descriptions_weather.csv")

    # parser.add_argument("--database_folder", type=str, default="/home/shared/datasets/msls_challenge")    
    # parser.add_argument("--queries_folder", type=str, default=None)       
    # parser.add_argument("--image_root", type=str, default="/home/shared/datasets/msls_challenge/test")    
    # parser.add_argument("--queries_csv", type=str, default="datasets/descriptions/msls_challenge_descriptions.csv")
    
    # parser.add_argument("--database_folder", type=str, default="/home/shared/datasets/nordland/test/database")    
    # parser.add_argument("--queries_folder", type=str, default="/home/shared/datasets/nordland/test/queries")       
    # parser.add_argument("--image_root", type=str, default="/home/shared/datasets/nordland/test")    
    # parser.add_argument("--queries_csv", type=str, default="datasets/descriptions/nordland_descriptions.csv")
    
    
    parser.add_argument("--num_workers", type=int, default=4, help="_")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="set to 1 if database images may have different resolution"
    )
    parser.add_argument(
        "--log_dir", type=str, default="default", help="experiment name, output logs will be saved under logs/log_dir"
    )
    parser.add_argument("--descriptor_dir", type=str, default="descriptors", help="_")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="_")
    parser.add_argument(
        "--recall_values",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
        help="values for recall (e.g. recall@1, recall@5)",
    )
    parser.add_argument(
        "--no_labels",
        action="store_true",
        help="set to true if you have no labels and just want to "
        "do standard image retrieval given two folders of queries and DB",
    )
    parser.add_argument(
        "--num_preds_to_save", type=int, default=0, help="set != 0 if you want to save predictions for each query"
    )
    parser.add_argument(
        "--save_only_wrong_preds",
        action="store_true",
        help="set to true if you want to save predictions only for " "wrongly predicted queries",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Resizing shape for images (HxW). If a single int is passed, set the"
        "smallest edge of all images to this value, while keeping aspect ratio",
    )
    parser.add_argument(
        "--save_descriptors",
        action="store_true",
        help="set to True if you want to save the descriptors extracted by the model",
    )
    parser.add_argument("--gpu", type=str, default="0", help="which gpu to use")
    parser.add_argument("--model_name", type=str, default='openai/clip-vit-base-patch16')
    parser.add_argument("--image_model_name", type=str, default='dinov2')
    parser.add_argument("--text_model_name", type=str, default='checkpoints/bge-l-lora-ms-merged')
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default=None)        
    parser.add_argument("--is_normalize", type=int, default="0", help="is normalize features")    
    parser.add_argument("--max_rerank", type=int, default="20", help="max results for reranking")        
    parser.add_argument("--train_vlm", type=int, default="1", help="train text encoder or not")
    parser.add_argument("--lora_all_linear", type=int, default="1", help="lora all linear 0=no/1=yes")
    parser.add_argument("--lora_target_modules", nargs='+', default=["query", "value"], help="when not lora_all_linear, lora target modules")    
    parser.add_argument("--lora_r", type=int, default="64", help="lora_all_linear 0=no/1=yes")     
    parser.add_argument("--agg_type", type=int, default="0", help="0=None, 1=mlp, 2=cosine, 3=2xcosine")
    parser.add_argument("--embeds_dim", type=int, default="512", help="embeds dimension")    
    parser.add_argument("--cross_modal", type=int, default="3", help="cross modal 0=no/1=clip/2=clip ft/3=lavpr_cross")        
    parser.add_argument("--reranker", type=int, default="0", help="reranker")        
    parser.add_argument("--reranker_mllm", type=int, default="0", help="reranker")        
    parser.add_argument("--reranker_filip", type=int, default="0", help="reranker")        
    parser.add_argument("--reranker_jev", type=int, default="0", help="reranker")        
    parser.add_argument("--rerank_alpha", type=float, default="0", help="0=reranker only, 1=global only, 0.5=equal mix")      
    parser.add_argument("--filip_retrieval", type=int, default="0", help="reranker")                  
    parser.add_argument("--bfloat16", type=int, default="0", help="bfloat16 or not")    
    parser.add_argument("--is_image", type=int, default="1", help="is image support")    
    parser.add_argument("--is_llp", type=int, default="0", help="is llp support")            
    parser.add_argument("--data", type=str, default=None, help="amstertime/pitts/msls")  
    parser.add_argument("--stage1_cache", type=str, default=None, help="stage1 cache")      
    parser.add_argument("--max_sim", type=int, default="0", help="is max sim reranking")  
    parser.add_argument('--rerank_internvl', type=str, default=None, help='Path to InternVL2 cross-encoder reranker checkpoint.')
    parser.add_argument('--internvl_batch', type=int, default=8, help='Batch size for InternVL2 reranking (memory intensive, keep small).')            

    args = parser.parse_args()
    
    args.use_labels = not args.no_labels
        
    return args
