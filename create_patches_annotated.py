import openslide
from PIL import Image
import numpy as np
import os
import cv2
import h5py
import csv
import concurrent.futures

def save_hdf5(output_path, patch_coords, metadata, mode='a'):
    with h5py.File(output_path, mode) as file:
        if 'coords' not in file:
            # Define the initial shape and max shape
            initial_shape = (len(patch_coords), 2)  # Assuming 2D coordinates (x, y)
            maxshape = (None, 2)
            dset = file.create_dataset('coords', shape=initial_shape, maxshape=maxshape, dtype='int64')
            dset[:] = patch_coords  # Assign all coordinates

            # Set attributes
            for key, value in metadata.items():
                dset.attrs[key] = value
        else:
            dset = file['coords']
            dset.resize(len(dset) + len(patch_coords), axis=0)
            dset[-len(patch_coords):] = patch_coords  # Assign new coordinates


def process_patch(slide, binary_mask, x, y, patch_size, is_40x, input_patch_size=256):
    patch = np.array(slide.read_region((x, y), 0, (patch_size, patch_size)).convert('RGB'))
    if is_40x:
        patch = np.array(Image.fromarray(patch).resize((input_patch_size, input_patch_size), Image.ANTIALIAS))
    
    mask_patch = binary_mask[y:y + input_patch_size, x:x + input_patch_size]
    empty_percentage = np.sum(np.array(Image.fromarray(patch).convert('L')) > 150) / float(patch.size / 3)

    if np.all(mask_patch == 255) and empty_percentage < 0.7:
        return (x, y), f'patch_{x}_{y}'
    return None, None

def extract_tumorous_patches(svs_path, mask_path, output_folder, input_patch_size=256, csv_file_path='processed_slides.csv'):
    slide = openslide.OpenSlide(svs_path)
    magnification = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, None)
    is_40x = magnification == '40'
    patch_size = 512 if is_40x else input_patch_size

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    os.makedirs(output_folder, exist_ok=True)
    resized_mask = cv2.resize(mask, slide.dimensions, interpolation=cv2.INTER_NEAREST)
    binary_mask = cv2.threshold(resized_mask, 1, 255, cv2.THRESH_BINARY)[1]

    h5_filename = os.path.join(output_folder, f'{os.path.splitext(os.path.basename(svs_path))[0]}.h5')
    tasks = []
    

    patch_coords = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for y in range(0, slide.dimensions[1], patch_size):
            for x in range(0, slide.dimensions[0], patch_size):
                tasks.append(executor.submit(process_patch, slide, binary_mask, x, y, patch_size, is_40x, input_patch_size))

        for future in concurrent.futures.as_completed(tasks):
            coords, patch_name = future.result()
            if coords is not None:
                patch_coords.append(coords)

    if patch_coords:
        metadata = {
            'downsample': [1.0, 1.0],
            'downsampled_level_dim': slide.level_dimensions[0],  
            'level_dim': slide.level_dimensions[0],
            'name': os.path.basename(svs_path),
            'patch_level': 0,
            'patch_size': input_patch_size,
            'save_path': output_folder
        }
        save_hdf5(h5_filename, np.array(patch_coords), metadata, mode='a')



    with open(csv_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([os.path.basename(svs_path), magnification])

    slide.close()
        
def process_directories(svs_directory, mask_directory, output_directory, mapping_csv):
    # Read the mapping from the CSV file
    mapping = {}
    with open(mapping_csv, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            mask_name, svs_name = row[0], row[1]
            mapping[mask_name] = svs_name

    # Iterate over all mask files
    for mask_filename in os.listdir(mask_directory):
        if mask_filename.lower().endswith('.png'):  # Adjust the extension if needed
            mask_basename = os.path.splitext(mask_filename)[0]
            if mask_basename in mapping:
                svs_filename = mapping[mask_basename]
                mask_path = os.path.join(mask_directory, mask_filename)
                svs_path = os.path.join(svs_directory, svs_filename)

                # Check if the corresponding SVS file exists
                if os.path.exists(svs_path):
                    extract_tumorous_patches(svs_path, mask_path, output_directory)
                else:
                    print(f"SVS file not found for mask {mask_filename}")
                    continue
            else:
                print(f"No mapping found for mask {mask_filename}")
                continue


if __name__ == "__main__":
    svs_directory = "/images/PublicDatasets/NSCLC/TCGA_Renal/TCGA_SVS_Collect_COMBINED"
    mask_directory = "/images/PublicDatasets/NSCLC/TCGA_Renal/RCC_Annotated_Patches/complete_region_annotation"
    output_directory = "/images/PublicDatasets/NSCLC/TCGA_Renal/RCC_Annotated_Patches/extracted_patches_annotated"
    mapping_csv = "/images/PublicDatasets/NSCLC/TCGA_Renal/RCC_Annotated_Patches/01_patch_annotated_list.csv"

    process_directories(svs_directory, mask_directory, output_directory, mapping_csv)
