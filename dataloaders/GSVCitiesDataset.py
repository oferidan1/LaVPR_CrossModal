# https://github.com/amaralibey/gsv-cities

import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
import posixpath
import json
import random
import ast

default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# NOTE: Hard coded path to dataset folder 
BASE_PATH = '/mnt/d/data/gsv_cities/'
TRAIN_CSV = '/mnt/d/data/gsv_cities/gsv_cities_predictions.csv'

SG_JSON = 'datasets/descriptions/gsv_cities_descriptions_sg.json'
ATTR_JSON = 'datasets/descriptions/gsv_cities_descriptions_attr_hn.json'


# if not Path(BASE_PATH).exists():
#     raise FileNotFoundError(
#         'BASE_PATH is hardcoded, please adjust to point to gsv_cities')

class GSVCitiesDataset(Dataset):
    def __init__(self,
                 cities=['London', 'Boston'],
                 img_per_place=4,
                 min_img_per_place=4,
                 random_sample_from_each_place=True,
                 transform=default_transform,
                 base_path=BASE_PATH,
                 train_csv=TRAIN_CSV,
                 mapping_json_path='datasets/gsv_cities_image_id_to_vocab_indices.json', 
                 max_concepts=64
                 ):
        super(GSVCitiesDataset, self).__init__()
        self.base_path = base_path
        self.train_csv = train_csv
        self.cities = cities

        assert img_per_place <= min_img_per_place, \
            f"img_per_place should be less than {min_img_per_place}"
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.random_sample_from_each_place = random_sample_from_each_place
        self.transform = transform
        
        # generate the dataframe contraining images metadata
        self.dataframe = self.__getdataframes()
        
        # get all unique place ids
        self.places_ids = pd.unique(self.dataframe.index)
        self.total_nb_images = len(self.dataframe)        
        
        self.image_path, self.image_full_path, self.description, self.flip_desc, self.hn_desc = GSVCitiesDataset.read_csv_file(train_csv, base_path)
        
        #load SG JSON
        # with open(SG_JSON, 'r') as f:
        #     sg_json = json.load(f)
            
        # # Transform the list into a fast-access scene graph dictionary
        # # This extracts 'scene_graph' directly so you don't have to type it out later
        # self.sg_json = {item["image_id"]: item["scene_graph"] for item in sg_json}
        
        #load ATTR JSON 
        # with open(ATTR_JSON, 'r') as f:
        #     self.attr_json = json.load(f)            
            
        # Load the precomputed cache mapping database
        with open(mapping_json_path, "r") as f:
            self.image_to_indices = json.load(f)
            
        self.max_concepts = max_concepts
        
    def __getdataframes(self):
        ''' 
            Return one dataframe containing
            all info about the images from all cities

            This requieres DataFrame files to be in a folder
            named Dataframes, containing a DataFrame
            for each city in self.cities
        '''
        # read the first city dataframe
        df = pd.read_csv(self.base_path+'Dataframes/'+f'{self.cities[0]}.csv')
        df = df.sample(frac=1)  # shuffle the city dataframe
        

        # append other cities one by one
        for i in range(1, len(self.cities)):
            tmp_df = pd.read_csv(
                self.base_path+'Dataframes/'+f'{self.cities[i]}.csv')

            # Now we add a prefix to place_id, so that we
            # don't confuse, say, place number 13 of NewYork
            # with place number 13 of London ==> (0000013 and 0500013)
            # We suppose that there is no city with more than
            # 99999 images and there won't be more than 99 cities
            # TODO: rename the dataset and hardcode these prefixes
            prefix = i
            tmp_df['place_id'] = tmp_df['place_id'] + (prefix * 10**5)
            tmp_df = tmp_df.sample(frac=1)  # shuffle the city dataframe
            
            df = pd.concat([df, tmp_df], ignore_index=True)

        # keep only places depicted by at least min_img_per_place images
        res = df[df.groupby('place_id')['place_id'].transform(
            'size') >= self.min_img_per_place]
        return res.set_index('place_id')
    
    def __getitem__(self, index):
        place_id = self.places_ids[index]
        
        # get the place in form of a dataframe (each row corresponds to one image)
        place = self.dataframe.loc[place_id]
        
        # sample K images (rows) from this place
        # we can either sort and take the most recent k images
        # or randomly sample them
        if self.random_sample_from_each_place:
            place = place.sample(n=self.img_per_place)
        else:  # always get the same most recent images
            place = place.sort_values(
                by=['year', 'month', 'lat'], ascending=False)
            place = place[: self.img_per_place]
            
        imgs = []
        descriptions = []
        flip_descs = []
        neg_attr_descs = []
        hn_descs = []
        concepts_ids = []
        #text_seg_per_place = []
        
        for i, row in place.iterrows():
            img_name = self.get_img_name(row)
            img_path = self.base_path + 'Images/' + \
                row['city_id'] + '/' + img_name
            img = self.image_loader(img_path)

            if self.transform is not None:
                img = self.transform(img)

            imgs.append(img)
            
            # get the description for this image
            # find image_path index in self.image_path  
            neg_desc, flip_desc, hn_desc, description = "", "", "", ""
            text_seg = []
            img_concepts_ids = torch.zeros(self.max_concepts, dtype=torch.long)
            if img_path in self.image_full_path:
                max_length = 256
                desc_index = self.image_full_path.index(img_path)                
                description = self.description[desc_index][:max_length]
                if self.flip_desc:
                    flip_desc = self.flip_desc[desc_index][:max_length]
                if self.hn_desc:
                    hn_desc = self.hn_desc[desc_index][:max_length]
                # if self.text_seg:
                #     text_seg = self.text_seg[desc_index]
                
                image_id = self.image_path[desc_index]                
                active_ids = self.image_to_indices.get(image_id, [])
                
                # 3. Construct fixed-length target tracking arrays padded with 0 (<PAD>)
                num_to_copy = min(len(active_ids), self.max_concepts)
                if num_to_copy > 0:
                    img_concepts_ids[:num_to_copy] = torch.tensor(active_ids[:num_to_copy], dtype=torch.long)
                
                #find image_id in sg_json
                # image_id = self.image_path[desc_index]
                # sg_data = self.sg_json[image_id]                
                # if sg_data:
                #     neg_desc = description.lower()
                #     objects = sg_data.get('objects', [])
                #     for obj in objects:
                #         obj_label = str(obj.get('label', obj.get('type', obj.get('name', '')))).lower().strip()
                #         obj_attributes = obj.get('attributes', [])
                #         if isinstance(obj_attributes, dict):
                #             obj_attributes = list(obj_attributes.values())
                        
                #         for attr in obj_attributes:          
                #             attr_val = attr.get('value', "") if isinstance(attr, dict) else attr
                #             if isinstance(attr_val, str) and attr_val.strip().startswith("[") and attr_val.strip().endswith("]"):
                #                 try:
                #                     attr_val = ast.literal_eval(attr_val.strip())
                #                 except (ValueError, SyntaxError):
                #                     pass
                #             if not isinstance(attr_val, list):
                #                 attr_val = [attr_val]
                #             for single_attr in attr_val:
                #                 attr_clean = str(single_attr).lower().strip()
                #                 if obj_label in self.attr_json and attr_clean in self.attr_json[obj_label]:
                #                     neg_attr = self.attr_json[obj_label][attr_clean]
                #                     if neg_attr:
                #                         neg = random.choice(neg_attr)
                #                         neg_desc = neg_desc.replace(attr_clean, neg)                            
                
            descriptions.append(description) 
            flip_descs.append(flip_desc)
            hn_descs.append(hn_desc)
            #neg_attr_descs.append(neg_desc)
            concepts_ids.append(img_concepts_ids)
            #text_seg_per_place.append(text_seg)
            
         # NOTE: contrary to image classification where __getitem__ returns only one image 
        # in GSVCities, we return a place, which is a Tesor of K images (K=self.img_per_place)
        # this will return a Tensor of shape [K, channels, height, width]. This needs to be taken into account 
        # in the Dataloader (which will yield batches of shape [BS, K, channels, height, width])
        return torch.stack(imgs), torch.tensor(place_id).repeat(self.img_per_place), descriptions, flip_descs, hn_descs, neg_attr_descs, torch.stack(concepts_ids)

    def __len__(self):
        '''Denotes the total number of places (not images)'''
        return len(self.places_ids)

    @staticmethod
    def image_loader(path):
        return Image.open(path).convert('RGB')

    @staticmethod
    def get_img_name(row):
        # given a row from the dataframe
        # return the corresponding image name

        city = row['city_id']
        
        # now remove the two digit we added to the id
        # they are superficially added to make ids different
        # for different cities
        pl_id = row.name % 10**5  #row.name is the index of the row, not to be confused with image name
        pl_id = str(pl_id).zfill(7)
        
        panoid = row['panoid']
        year = str(row['year']).zfill(4)
        month = str(row['month']).zfill(2)
        northdeg = str(row['northdeg']).zfill(3)
        lat, lon = str(row['lat']), str(row['lon'])
        name = city+'_'+pl_id+'_'+year+'_'+month+'_' + \
            northdeg+'_'+lat+'_'+lon+'_'+panoid+'.jpg'
        return name

    @staticmethod
    def read_csv_file(labels_file, image_root):    
        df = pd.read_csv(labels_file, 
            engine='python',  # Use python engine for better path handling
            encoding='utf-8',
            on_bad_lines='skip',
            quotechar='"',
            skipinitialspace=True)
        flip_desc = None
        hn_desc = None
        image_path = df['image_path'].values
        description = df['description'].values    
        # if 'flip' in df.columns:
        #     flip_desc = df['flip'].values                    
        if 'hn' in df.columns:
            hn_desc = df['hn'].values    
        # if 'compressed_description' in df.columns:
        #     description = df['compressed_description'].values       
        flip_desc_ = [flip_text_by_comma(t) for t in description]        
        # text_seg = [
        #     [seg.strip() for seg in str(text).split(',') if seg.strip()] or [str(text)]
        #     for text in description
        # ]
        image_full_path = [posixpath.join(image_root, p) for p in image_path]        
        return image_path, image_full_path, description, flip_desc, hn_desc#, text_seg

   
@staticmethod
def flip_text_by_comma(text):
    phrases = [p.strip() for p in text.split(',')]
    return ', '.join(phrases[::-1])

class CachedVPRSGDataset(Dataset):
    """
    High-performance VPR dataset utilizing precomputed scene graph 
    vocab indices to eliminate training-time string processing bottle-necks.
    """
    def __init__(self, dataset_list_path, mapping_json_path, max_concepts=64, transform=None):
        super().__init__()
        # Load your standard training image array split configuration
        with open(dataset_list_path, "r") as f:
            self.dataset = json.load(f)
            
        # Load the precomputed cache mapping database
        with open(mapping_json_path, "r") as f:
            self.image_to_indices = json.load(f)
            
        self.max_concepts = max_concepts
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        entry = self.dataset[idx]
        image_id = entry.get("image_id")
        
        # 1. Load image tensor via standard transformations pipeline
        # img_path = os.path.join(self.images_root, image_id)
        # image = Image.open(img_path).convert("RGB")
        # if self.transform: image = self.transform(image)
        image_tensor = torch.zeros(3, 224, 224, dtype=torch.float32) # Placeholder
        
        # 2. Extract cached active integer category index pointers
        active_ids = self.image_to_indices.get(image_id, [])
        
        # 3. Construct fixed-length target tracking arrays padded with 0 (<PAD>)
        padded_ids = torch.zeros(self.max_concepts, dtype=torch.long)
        num_to_copy = min(len(active_ids), self.max_concepts)
        if num_to_copy > 0:
            padded_ids[:num_to_copy] = torch.tensor(active_ids[:num_to_copy], dtype=torch.long)
            
        return {
            "image": image_tensor,
            "concept_ids": padded_ids,
            "image_id": image_id
        }