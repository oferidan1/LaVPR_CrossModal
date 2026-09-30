import posixpath
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors


def parse_utm_coords(path_series):
    """
    Extracts UTM easting and northing coordinates from paths formatted as:
    path/to/file/@utm_east@utm_north@...@.jpg
    """
    utms = []
    valid_indices = []
    
    for idx, path in enumerate(path_series):
        try:
            parts = str(path).split("@")
            east = float(parts[1])
            north = float(parts[2])
            utms.append((east, north))
            valid_indices.append(idx)
        except (IndexError, ValueError):
            continue
            
    return np.array(utms), valid_indices


def cluster_paths_to_place_ids(paths, utms, start_id=0, radius=25.0):
    """
    Clusters UTM coordinates within radius (meters) and assigns 
    numeric integer place_ids starting from `start_id`.
    """
    if len(utms) == 0:
        return {}, start_id

    knn = NearestNeighbors(n_jobs=-1)
    knn.fit(utms)
    _, neighbors = knn.radius_neighbors(utms, radius=radius)

    G = nx.Graph()
    for p in paths:
        G.add_node(p)

    for i, neighbor_indices in enumerate(neighbors):
        src_path = paths[i]
        for target_idx in neighbor_indices:
            G.add_edge(src_path, paths[target_idx])

    path_to_place_id = {}
    current_id = start_id
    for component in nx.connected_components(G):
        for img_path in component:
            path_to_place_id[img_path] = current_id
        current_id += 1

    return path_to_place_id, current_id


def process_csv_and_add_place_ids(
    csv_path,
    output_csv_path,
    image_root="",
    database_folder="database",
    queries_folder="queries",
    radius=25.0,
    cluster_together=True  # True: Unified IDs across Query & DB | False: Separated IDs
):
    print(f"Reading original CSV from: {csv_path}")
    df = pd.read_csv(
        csv_path,
        engine="python",
        encoding="utf-8",
        on_bad_lines="skip",
        quotechar='"',
        skipinitialspace=True,
    )

    if image_root:
        full_paths = [posixpath.join(image_root, p) for p in df["image_path"].values]
    else:
        full_paths = df["image_path"].values

    df["_resolved_path"] = full_paths
    path_to_place_id = {}

    if cluster_together:
        print(f"Clustering ALL images together into unified numeric place_ids (radius <= {radius}m)...")
        utms, valid_indices = parse_utm_coords(df["_resolved_path"].values)
        valid_paths = df["_resolved_path"].iloc[valid_indices].values

        path_to_place_id, _ = cluster_paths_to_place_ids(
            valid_paths, utms, start_id=0, radius=radius
        )
    else:
        print(f"Clustering QUERY and DATABASE images SEPARATELY (radius <= {radius}m)...")
        db_mask = df["_resolved_path"].str.contains(database_folder, na=False)
        q_mask = df["_resolved_path"].str.contains(queries_folder, na=False)

        # 1. Cluster Database Images
        db_paths = df.loc[db_mask, "_resolved_path"].values
        db_utms, db_valid_idx = parse_utm_coords(db_paths)
        db_valid_paths = db_paths[db_valid_idx]

        db_place_map, next_id = cluster_paths_to_place_ids(
            db_valid_paths, db_utms, start_id=0, radius=radius
        )
        path_to_place_id.update(db_place_map)

        # 2. Cluster Query Images (Continues numeric IDs from next_id to avoid collision)
        q_paths = df.loc[q_mask, "_resolved_path"].values
        q_utms, q_valid_idx = parse_utm_coords(q_paths)
        q_valid_paths = q_paths[q_valid_idx]

        q_place_map, _ = cluster_paths_to_place_ids(
            q_valid_paths, q_utms, start_id=next_id, radius=radius
        )
        path_to_place_id.update(q_place_map)

    # Assign integer place_id back to DataFrame
    max_id = max(path_to_place_id.values()) if path_to_place_id else -1
    unclustered_counter = max_id + 1

    final_place_ids = []
    for p in df["_resolved_path"].values:
        if p in path_to_place_id:
            final_place_ids.append(path_to_place_id[p])
        else:
            final_place_ids.append(unclustered_counter)
            unclustered_counter += 1

    df["place_id"] = final_place_ids

    # Clean up temporary resolved path column
    df.drop(columns=["_resolved_path"], inplace=True)

    print(f"Assigned {df['place_id'].nunique()} unique numeric place_ids across {len(df)} rows.")

    # Save to output file
    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"Successfully saved updated CSV with numeric place_ids to: {output_csv_path}")


if __name__ == "__main__":
    CSV_PATH = "datasets/descriptions/amstertime_descriptions.csv"
    OUTPUT_CSV_PATH = "datasets/descriptions/amstertime_descriptions_with_place_id.csv"
    IMAGE_ROOT = "/home/shared/datasets/amstertime/test/"
    DATABASE_FOLDER = "database"
    QUERIES_FOLDER = "queries"
    
    cluster_together = False
    
    process_csv_and_add_place_ids(
        csv_path=CSV_PATH,
        output_csv_path=OUTPUT_CSV_PATH,
        image_root=IMAGE_ROOT,
        database_folder=DATABASE_FOLDER,
        queries_folder=QUERIES_FOLDER,
        radius=25.0,
        cluster_together=cluster_together  # Set to False to separate query & db cluster spaces
    )
