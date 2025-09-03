"""
This file (dump_dataset.py) is designed for:
    dump dataset for DMD evaluating
Copyright (c) 2024, Zhiyu Pan. All rights reserved.
"""
import os
import os.path as osp
import pickle
import random
import numpy as np
import argparse
import pandas as pd # Import pandas for CSV loading

area_thresh = 40000

def load_minutiae_from_mnt(mnt_file_path):
    """Loads minutiae from a .mnt file (CSV-like format)."""
    try:
        mnt_df = pd.read_csv(mnt_file_path)
        # Assuming columns are 'Unnamed: 0', 'x', 'y', 'angle', 'score', 'ridge_type'
        mnts = mnt_df[['x', 'y', 'angle']].values # Select x, y, angle columns
        return mnts
    except Exception as e:
        print(f"Error loading minutiae from {mnt_file_path}: {e}")
        return np.empty((0, 3)) # Return empty array in case of error


def create_datalist(prefix, dataname, img_type='png'): # default img_type to png
    # TEST_DATA dataset
    img_lst = []
    anchor_2d = []
    mnt_folder = osp.join(prefix, 'mnt') # path to mnt folder
    mnt_gallery_folder = osp.join(mnt_folder, 'gallery')
    mnt_query_folder = osp.join(mnt_folder, 'query')
    mnt_gallery_files = os.listdir(mnt_gallery_folder)
    mnt_query_files = os.listdir(mnt_query_folder)

    for mnt_f in mnt_gallery_files:
        mnt_path = osp.join(mnt_gallery_folder, mnt_f)
        mnts = load_minutiae_from_mnt(mnt_path) # Use custom loading function
        if mnts.size > 0: # Check if minutiae were loaded successfully
            for mnt_ in mnts: # one mnt per sample
                img_name = mnt_f.replace('.mnt', f'.{img_type}')
                img_lst.append(osp.join(dataname, "image", 'gallery', img_name)) # Construct image path relative to dataset root
                anchor_2d.append(mnt_)

    for mnt_f in mnt_query_files:
        mnt_path = osp.join(mnt_query_folder, mnt_f)
        mnts = load_minutiae_from_mnt(mnt_path) # Use custom loading function
        if mnts.size > 0: # Check if minutiae were loaded successfully
            for mnt_ in mnts:
                img_name = mnt_f.replace('.mnt', f'.{img_type}')
                img_lst.append(osp.join(dataname, "image", 'query', img_name)) # Construct image path relative to dataset root
                anchor_2d.append(mnt_)

    data_lst = {"img": img_lst,  "pose_2d": anchor_2d}
    print(f'{dataname} total {len(img_lst)} samples')
    return data_lst

if __name__ == "__main__":
    random.seed(1016)
    np.random.seed(1016)
    parser = argparse.ArgumentParser("Evaluation for DMD")
    parser.add_argument("--prefix", type=str, default="TEST_DATA/NIST27") # Adjusted prefix path
    args = parser.parse_args()

    # List of datasets to process and their corresponding image types
    # In the original create_pkl.py, this was:
    # datasets_to_process_names = ['sample_finger_enh_afis']
    # image_extensions = ['png']
    # We can combine them for clarity if processing multiple types in the future
    datasets_config = [
        ('NIST27', 'png')
    ]

    output_dir = './datasets'
    
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    for dataset_name, img_extension in datasets_config:
        # The 'args.prefix' is the base path for the current dataset_name's files.
        # 'dataset_name' is used for constructing image paths within the .pkl file
        # and for naming the output .pkl file.
        datalist_dict_of_lists = create_datalist(args.prefix, dataset_name, img_extension)
        
        # Convert the dictionary of lists to a list of dictionaries
        # This structure matches the per-dataset output of dump_dataset_mnteval.py
        processed_datalist = [dict(zip(datalist_dict_of_lists, v)) for v in zip(*datalist_dict_of_lists.values())]
        
        # Define the save file name based on the dataset name, similar to dump_dataset_mnteval.py
        save_file_path = osp.join(output_dir, f'{dataset_name}.pkl')
        
        # Save the data for the current dataset
        with open(save_file_path, "wb") as fp:
            pickle.dump(processed_datalist, fp) # Save the list of dictionaries directly
        
        print(f"Dataset '{dataset_name}' saved to: {save_file_path}")

    print("Done!")