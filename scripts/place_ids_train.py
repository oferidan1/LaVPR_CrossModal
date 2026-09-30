import posixpath
import pandas as pd
from pathlib import Path

# NOTE: Set your file paths here
BASE_PATH = '/home/shared/datasets/gsv_cities/'
TRAIN_CSV = 'datasets/descriptions/gsv_cities_descriptions.csv'
OUTPUT_CSV = 'gsv_cities_predictions_with_place_id.csv'

def get_img_name(row):
    """Generates the standardized GSV image filename from DataFrame metadata."""
    city = row['city_id']
    pl_id = str(row['place_id']).zfill(7)
    panoid = row['panoid']
    year = str(row['year']).zfill(4)
    month = str(row['month']).zfill(2)
    northdeg = str(row['northdeg']).zfill(3)
    lat, lon = str(row['lat']), str(row['lon'])

    return f"{city}_{pl_id}_{year}_{month}_{northdeg}_{lat}_{lon}_{panoid}.jpg"


def match_and_save_csv():
    dataframes_dir = Path(BASE_PATH) / 'Dataframes'
    csv_files = sorted(list(dataframes_dir.glob('*.csv')))

    print(f"1/4 Found {len(csv_files)} city dataframes. Building filename lookup table...")
    
    filename_to_place = {}
    
    # Process each city CSV independently using its raw place_id
    for i, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        
        # Apply city prefix matching GSVCitiesDataset conventions
        df['global_place_id'] = df['place_id'] + (i * 10**5)
        
        for _, row in df.iterrows():
            img_name = get_img_name(row)
            # Store filename -> global_place_id mapping
            filename_to_place[img_name] = row['global_place_id']

    print("2/4 Reading predictions CSV file...")
    predictions_df = pd.read_csv(
        TRAIN_CSV,
        engine='python',
        encoding='utf-8',
        on_bad_lines='skip',
        quotechar='"',
        skipinitialspace=True,
    )

    print("3/4 Matching by image filenames...")
    # Extract only the base filename (e.g. 'London_0000001_...jpg') to bypass root path differences
    extracted_filenames = [Path(p).name for p in predictions_df['image_path']]
    
    predictions_df['place_id'] = [
        filename_to_place.get(fname, -1) for fname in extracted_filenames
    ]

    matched_count = (predictions_df['place_id'] != -1).sum()
    total_count = len(predictions_df)
    print(f"Matched {matched_count} / {total_count} images ({matched_count/total_count:.2%})")

    print("4/4 Saving matched output to CSV...")
    predictions_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Successfully saved to: {OUTPUT_CSV}")


if __name__ == '__main__':
    match_and_save_csv()