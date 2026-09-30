import math
import os
import re
import time
from collections import Counter
from joblib import Parallel, delayed
import pandas as pd
import spacy

# Load spaCy model globally
nlp = spacy.load(
    "en_core_web_sm", disable=["tok2vec", "parser", "ner", "senter"]
)

# Generic urban VPR background terms to penalize
GENERIC_VPR_WORDS = {
    "road",
    "street",
    "sidewalk",
    "pavement",
    "asphalt",
    "tree",
    "trees",
    "foliage",
    "sky",
    "cloud",
    "clouds",
    "building",
    "facade",
    "wall",
    "pole",
    "view",
    "scene",
    "background",
    "foreground",
    "mid-distance",
    "distance",
    "part",
    "portion",
    "section",
    "side",
    "curb",
    "grass",
    "grassy",
    "bush",
    "bushes",
    "hedge",
    "fence",
    "window",
    "windows",
    "door",
    "entrance",
    "structure",
    "area",
}


def read_csv_file(labels_file):
  """User-provided CSV reader."""
  df = pd.read_csv(
      labels_file,
      engine="python",  # Use python engine for better path handling
      encoding="utf-8",
      on_bad_lines="skip",
      quotechar='"',
      skipinitialspace=True,
  )
  image_path = df["image_path"].values
  description = df["description"].values
  return df, image_path, description


def get_words_and_lemmas(text):
  """Extracts lowercase non-stopword lemmas."""
  if not isinstance(text, str):
    return []
  doc = nlp(text.lower())
  return [
      token.lemma_
      for token in doc
      if token.is_alpha and not token.is_stop and len(token.lemma_) > 1
  ]


def extract_smart_chunks(text):
  """Extracts quotes first (OCR/signs), then splits on clause boundaries."""
  if not isinstance(text, str) or not text.strip():
    return []

  chunks = []
  # Extract quoted text (highest priority for signs/OCR)
  quotes = re.findall(r'"([^"]+)"', text)
  for q in quotes:
    if len(q.strip()) > 2:
      chunks.append(f'"{q.strip()}"')

  clean_text = re.sub(r'"[^"]+"', "", text)
  raw_splits = re.split(
      r"[;:]|\b(?:and|then|followed by|featuring|with a)\b", clean_text
  )

  for split in raw_splits:
    for sub in split.split(","):
      c = sub.strip()
      if c and len(c) > 3:
        chunks.append(c)

  return chunks


def score_chunk_strict(chunk, lemma_idf):
  """Scores a chunk based on rare landmark words while penalizing generic background terms."""
  lemmas = get_words_and_lemmas(chunk)
  if not lemmas:
    return 0.0

  rare_scores = []
  generic_count = 0

  for lem in lemmas:
    if lem in GENERIC_VPR_WORDS:
      generic_count += 1
    else:
      rare_scores.append(lemma_idf.get(lem, 1.0))

  # Quoted OCR text gets top priority
  if chunk.startswith('"') and chunk.endswith('"'):
    return 10.0

  if not rare_scores:
    return 0.0

  # Max rare IDF penalized by generic word density
  return max(rare_scores) / (1.0 + 0.8 * generic_count)


def process_single_description(
    description, lemma_idf, score_threshold, max_chunks_per_image
):
  """Processes an individual image caption."""
  chunks = extract_smart_chunks(description)
  if not chunks:
    return ""

  scored_chunks = []
  for chunk in chunks:
    score = score_chunk_strict(chunk, lemma_idf)
    scored_chunks.append((chunk, score))

  scored_chunks.sort(key=lambda x: x[1], reverse=True)

  selected = []
  seen = set()
  for chunk_text, score in scored_chunks:
    norm = chunk_text.lower()
    if score >= score_threshold and norm not in seen:
      seen.add(norm)
      selected.append(chunk_text)
    if len(selected) == max_chunks_per_image:
      break

  # Fallback: keep the single best chunk if all were below threshold
  if not selected and scored_chunks:
    selected = [scored_chunks[0][0]]

  return ", ".join(selected)


def compress_csv_dataset(
    input_csv_file,
    output_csv_file,
    score_threshold=2.5,
    max_chunks_per_image=3,
    n_jobs=-1,
):
  start_time = time.time()
  print(f"Reading CSV file: {input_csv_file}")

  # 1. Read CSV using your function
  df, image_paths, descriptions = read_csv_file(input_csv_file)
  N = len(descriptions)
  print(f"Loaded {N:,} rows successfully.")

  # 2. Extract Chunks & Compute Lemma Document Frequencies
  print("Extracting chunks and computing corpus-wide Document Frequency...")
  all_extracted_chunks = []
  lemma_df_counts = Counter()

  for idx, desc in enumerate(descriptions):
    chunks = extract_smart_chunks(desc)
    all_extracted_chunks.append(chunks)

    full_text = " ".join(chunks)
    lemmas = set(get_words_and_lemmas(full_text))
    lemma_df_counts.update(lemmas)

    if (idx + 1) % 100000 == 0:
      print(f"  Processed {idx + 1:,} / {N:,} captions...")

  # 3. Calculate IDF Scores
  print("Calculating Word IDF scores...")
  lemma_idf = {
      lemma: math.log((N + 1) / (df + 1)) + 1
      for lemma, df in lemma_df_counts.items()
  }

  # 4. Process all descriptions in parallel across CPU cores
  print(f"Compressing descriptions in parallel using {n_jobs} cores...")

  def worker(desc):
    return process_single_description(
        desc, lemma_idf, score_threshold, max_chunks_per_image
    )

  compressed_descriptions = Parallel(n_jobs=n_jobs, batch_size=2000)(
      delayed(worker)(desc) for desc in descriptions
  )

  # 5. Append new column and save CSV in identical format
  df["compressed_description"] = compressed_descriptions

  print(f"Saving updated CSV to: {output_csv_file}")
  df.to_csv(output_csv_file, index=False, encoding="utf-8", quoting=1)

  elapsed = time.time() - start_time
  print(
      f"=== Successfully processed {N:,} rows in {elapsed / 60:.2f} minutes ==="
  )


# --- Execution Entry Point ---
if __name__ == "__main__":
  #input_csv_path = "datasets/descriptions/gsv_cities_descriptions.csv"  # Path to your input CSV
  input_csv_path = "datasets/descriptions/pitts30k_test_descriptions.csv"  # Path to your input CSV
  output_csv_path = "lavpr_dataset_compressed.csv"  # Path to save updated CSV

  if os.path.exists(input_csv_path):
    compress_csv_dataset(
        input_csv_file=input_csv_path,
        output_csv_file=output_csv_path,
        score_threshold=2.5,
        max_chunks_per_image=6,
        n_jobs=-1,  # Uses all available CPU cores
    )
  else:
    print(f"Input file '{input_csv_path}' not found. Please verify the path.")