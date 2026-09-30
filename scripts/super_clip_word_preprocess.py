import json
import os
import csv
import re
import argparse
import torch
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from nltk.stem import WordNetLemmatizer
from difflib import SequenceMatcher
import nltk

def bootstrap_nltk_dependencies():
    """Ensures all lexical, morphological, and POS tagging resources are available."""
    requirements = ['words', 'stopwords', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger']
    for dependency in requirements:
        try:
            if dependency == 'words':
                nltk.data.find('corpora/words')
            elif dependency == 'averaged_perceptron_tagger':
                nltk.data.find('taggers/averaged_perceptron_tagger')
            else:
                nltk.data.find(f"corpora/{dependency}")
        except LookupError:
            print(f"Downloading required NLTK resource: '{dependency}'...")
            nltk.download(dependency, quiet=True)

# Run bootstrap sequence before importing corpus maps
bootstrap_nltk_dependencies()
from nltk.corpus import words as nltk_words
from nltk.corpus import stopwords as nltk_stopwords
from nltk import pos_tag

# Global corpus infrastructure for O(1) validations
ENGLISH_VOCAB = set(w.lower() for w in nltk_words.words())
STOPWORDS = set(nltk_stopwords.words('english'))

# Valid architectural and visual primitives under 4 characters
VALID_SHORT_WORDS = {
    'atm', 'bus', 'car', 'van', 'bin', 'bay', 'bed', 'arc', 'elm', 'oak', 'pub', 'bar', 
    'sub', 'sky', 'dot', 'box', 'ray', 'oil', 'cab', 'cap', 'map', 'law', 'tag', 'hut', 'gym'
}

# Advanced cross-lingual mapper to collapse foreign storefront leakage into shared English targets
HARD_CONCEPT_COLLAPSER = {
    'abad': 'abbey', 'abada': 'abbey', 'abade': 'abbey', 'abadia': 'abbey',
    'aberto': 'open', 'aberta': 'open', 'abertos': 'open', 'abertas': 'open', 'abertura': 'open',
    'abierto': 'open', 'abierta': 'open', 'abrimos': 'open',
    'abastecedora': 'supplier', 'abarrotes': 'grocery', 'abogado': 'lawyer', 'abogada': 'lawyer',
    'accesorios': 'accessory', 'accessoires': 'accessory', 'accessori': 'accessory', 'accesories': 'accessory',
    'acondicionado': 'air', 'establishment': 'shop', 'estabelecimento': 'shop', 'abbigliamento': 'clothing',
    'abiti': 'clothing', 'abril': 'april', 'abbod': 'abbot',
    'acab': 'finish', 'accion': 'action', 'acesso': 'access', 'acess': 'access',
    'achat': 'buy', 'achetons': 'buy', 'acier': 'steel', 'acion': 'action', 'acuerdo': 'agreement',
    'transition': 'transition', 'transitional': 'transition', 'translational': 'transition'
}

def is_garbage_token(w):
    """
    Evaluates tokens against structural noise regularities, character repetitions, 
    and alternating/double-stutter VLM generation loops.
    """
    # 1. Catch consecutive repeating characters (e.g., 'aaaaa', 'wwwwww')
    if re.search(r'(.)\1{2,}', w):
        return True
        
    # 2. Catch alternating VLM decoding loops (e.g., 'ayayayaya', 'xoxoxoxo', 'ababab')
    if re.search(r'(.{1,2})\1{2,}', w):
        return True
        
    # 3. Guard Layer: Catch double-stutter VLM token faults (e.g., 'aabbitt', 'ccbbyy')
    if re.search(r'(.)\1.*(.)\2', w) and w not in ENGLISH_VOCAB:
        return True
        
    # 4. Filter out random structural string sequences missing vowels entirely
    vowels = set("aeiouy")
    if not any(char in vowels for char in w) and len(w) > 3:
        return True
        
    # 5. Suppress massive squashed compound text/URLs that pass the digit filter
    if len(w) > 13 and w not in ENGLISH_VOCAB:
        return True
        
    return False

def process_description_tokens(description, lemmatizer):
    """
    Parses sentences preserving mixed-case structures for accurate POS tagging,
    extracts target landmark proper nouns, and enforces English common nouns.
    """
    tokens = [t for t in re.split(r'[^a-zA-Z]+', description) if t]
    if not tokens:
        return []
        
    tagged_tokens = pos_tag(tokens)
    valid_sentence_words = []
    
    for raw_word, tag in tagged_tokens:
        w = raw_word.lower().strip()
        
        if not w or w.isdigit():
            continue
            
        # Hard Filter: Enforce strict minimum length of 4 characters to destroy short OCR fragments
        if len(w) < 4 and w not in VALID_SHORT_WORDS:
            continue
            
        if is_garbage_token(w):
            continue
            
        if w in HARD_CONCEPT_COLLAPSER:
            w = HARD_CONCEPT_COLLAPSER[w]
            
        lemma = lemmatizer.lemmatize(w)
        
        if lemma in STOPWORDS:
            continue
            
        # Proper nouns bypass the strict dictionary check to keep high-value landmarks
        is_proper_noun = tag in ('NNP', 'NNPS')
        
        if lemma not in ENGLISH_VOCAB and lemma not in VALID_SHORT_WORDS:
            if not is_proper_noun:
                continue  
                
        valid_sentence_words.append(lemma)
        
    return valid_sentence_words

def resolve_lexical_variants(word_list):
    """
    Two-pass contextual scanner that shrinks morphological derivations to short valid bases
    and lifts truncated text fragments up to their complete dictionary forms.
    """
    sorted_words = sorted(list(word_list), key=len)
    variant_map = {}
    
    VALID_SUFFIXES = {'al', 'nal', 'tional', 'l', 'el', 'ble', 'ive', 'ing', 'ed', 's', 'ment', 'ion', 'ate', 'ation', 'or'}
    
    # Pass 1: Shrink long morphological variants down to target short roots (transitional -> transition)
    for i, long_w in enumerate(sorted_words):
        for short_w in sorted_words[:i]:
            if len(short_w) < 4:
                continue
            if long_w.startswith(short_w):
                suffix = long_w[len(short_w):]
                if suffix in VALID_SUFFIXES or short_w in ENGLISH_VOCAB:
                    ratio = SequenceMatcher(None, short_w, long_w).ratio()
                    if ratio >= 0.70:
                        variant_map[long_w] = short_w
                        break
                        
    # Pass 2: Lift broken fragments up to full target dictionary entries (acceler -> accelerate)
    for i, long_w in enumerate(sorted_words):
        if long_w not in ENGLISH_VOCAB:
            continue
        for short_w in sorted_words[:i]:
            if len(short_w) < 4 or short_w in ENGLISH_VOCAB:
                continue
            if long_w.startswith(short_w):
                ratio = SequenceMatcher(None, short_w, long_w).ratio()
                if ratio >= 0.70:
                    variant_map[short_w] = long_w
                    
    return variant_map

def build_word_semantic_lookup(unique_words, model, distance_threshold):
    """Clusters validated vocabulary terms into dense embedding semantic groupings."""
    word_list = sorted(list(unique_words))
    if not word_list:
        return {}
    if len(word_list) == 1:
        return {word_list[0]: word_list[0]}

    embeddings = model.encode(word_list, show_progress_bar=False, convert_to_numpy=True)
    embeddings = normalize(embeddings)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="euclidean",
        linkage="average",
        distance_threshold=distance_threshold
    )
    labels = clustering.fit_predict(embeddings)

    cluster_to_words = defaultdict(list)
    for word, c_id in zip(word_list, labels):
        cluster_to_words[c_id].append(word)

    word_to_canonical = {}
    for c_id, words in cluster_to_words.items():
        # Optimization Layer: Protect cluster from fragment collapse by forcing dictionary targets
        dict_words = [w for w in words if w in ENGLISH_VOCAB]
        
        if dict_words:
            canonical_root = min(dict_words, key=len)
        else:
            long_enough_words = [w for w in words if len(w) >= 4]
            if long_enough_words:
                canonical_root = min(long_enough_words, key=len)
            else:
                canonical_root = min(words, key=len)
                
        for raw_word in words:
            word_to_canonical[raw_word] = canonical_root
            
    return word_to_canonical

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, default="datasets/descriptions/gsv_cities_descriptions.csv")
    parser.add_argument("--out_vocab_file", type=str, default="scene_graph_vocab.json")
    parser.add_argument("--out_img_idf", type=str, default="gsv_cities_image_idf.pt")
    parser.add_argument("--out_img_map", type=str, default="image_id_to_vocab_indices.json")
    parser.add_argument("--word_threshold", type=float, default=0.60)
    parser.add_argument("--min_freq", type=int, default=3, help="Minimum count threshold for validated landmarks")
    
    args = parser.parse_args()
    lemmatizer = WordNetLemmatizer()

    print(f"Opening full CSV database file from: {args.csv_file}")
    raw_word_pool = set()
    csv_rows_cache = []
    image_to_extracted_words = []

    with open(args.csv_file, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)  
        
        for row in reader:
            if len(row) < 2: continue
            img_path, description = row[0].strip(), row[1].strip()
            
            cleaned_entry_words = process_description_tokens(description, lemmatizer)
            
            for word in cleaned_entry_words:
                raw_word_pool.add(word)
                
            csv_rows_cache.append((img_path, description))
            image_to_extracted_words.append(set(cleaned_entry_words))

    print(f" -> Raw text isolation completed. Pool contains {len(raw_word_pool)} unique base tokens.")

    print("Executing structural and morphological variant aggregation mapping...")
    lexical_lookup_map = resolve_lexical_variants(raw_word_pool)
    
    consolidated_word_pool = set()
    image_to_consolidated_words = []
    
    for img_set in image_to_extracted_words:
        updated_set = set()
        for w in img_set:
            # Map elements through concept transformations and variants
            base_w = HARD_CONCEPT_COLLAPSER.get(w, w)
            final_word = lexical_lookup_map.get(base_w, base_w)
            updated_set.add(final_word)
            consolidated_word_pool.add(final_word)
        image_to_consolidated_words.append(updated_set)

    print("Initializing Sentence-Transformer model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    word_to_canonical = build_word_semantic_lookup(consolidated_word_pool, model, args.word_threshold)

    print("Accumulating global frequencies over normalized semantic landmarks...")
    canonical_word_frequencies = defaultdict(int)
    image_to_canonical_words = []

    for img_set in image_to_consolidated_words:
        canonical_set = set()
        for w in img_set:
            canonical_word = word_to_canonical.get(w, w)
            canonical_set.add(canonical_word)
            
        for c_word in canonical_set:
            canonical_word_frequencies[c_word] += 1
        image_to_canonical_words.append(canonical_set)

    filtered_vocab_words = [
        w for w in canonical_word_frequencies 
        if canonical_word_frequencies[w] >= args.min_freq
    ]
    
    sorted_vocab = sorted(filtered_vocab_words)
    concept_vocab = {"<PAD>": 0}
    for idx, word in enumerate(sorted_vocab):
        concept_vocab[word] = idx + 1
        
    print(f"Generated clean multi-label target vocabulary. Size (M) = {len(concept_vocab)} unique words.")
    
    with open(args.out_vocab_file, "w") as f:
        json.dump(concept_vocab, f, indent=4)

    print("Vectorizing global matrix documents counts and compiling lookup maps...")
    image_counts = torch.zeros(len(concept_vocab), dtype=torch.float32)
    image_id_to_vocab_indices = {}

    for (img_path, _), canonical_set in zip(csv_rows_cache, image_to_canonical_words):
        active_indices = list(set([concept_vocab[w] for w in canonical_set if w in concept_vocab]))
        if active_indices:
            image_counts[active_indices] += 1
        if img_path:
            image_id_to_vocab_indices[img_path] = sorted(active_indices)

    with open(args.out_img_map, "w") as f:
        json.dump(image_id_to_vocab_indices, f, indent=4)

    idf_image = torch.log(torch.tensor(len(csv_rows_cache)) / (1.0 + image_counts))
    idf_image[0] = 0.0
    idf_image.clamp_(min=0.0)
    
    torch.save(idf_image, args.out_img_idf)
    print("Vocab normalization pipeline executed successfully!")

if __name__ == "__main__":
    main()