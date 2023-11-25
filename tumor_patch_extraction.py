
import openslide
from PIL import Image
import numpy as np
import os
import cv2

def extract_tumorous_patches(svs_path, mask_path, output_folder, patch_size, empty_threshold=0.7):
    # Open the Whole Slide Image
    slide = openslide.OpenSlide(svs_path)

    # Open the segmentation mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get dimensions of the Whole Slide Image
    slide_width, slide_height = slide.dimensions

    # Get dimensions of the segmentation mask
    mask_width, mask_height = mask.shape[::-1]

    # Calculate the scaling factors
    width_scale = slide_width / mask_width
    height_scale = slide_height / mask_height

    # Rescale the mask to match the dimensions of the Whole Slide Image
    resized_mask = cv2.resize(mask, (slide_width, slide_height), interpolation=cv2.INTER_NEAREST)

    # Convert the mask to a binary mask where tumorous regions are white and non-tumorous regions are black
    binary_mask = cv2.threshold(resized_mask, 1, 255, cv2.THRESH_BINARY)[1]

    # Iterate through patches
    for y in range(0, slide_height, patch_size):
        for x in range(0, slide_width, patch_size):
            # Extract patch from Whole Slide Image
            #patch = np.array(slide.read_region((x, y), 0, (patch_size, patch_size)))
            #patch_grey_scale = patch.convert('L')
            patch_greyscale = np.array(slide.read_region((x, y), 0, (patch_size, patch_size)).convert('L'))
            patch = np.array(slide.read_region((x, y), 0, (patch_size, patch_size)))

            # Extract corresponding region from binary mask
            mask_patch = binary_mask[y:y+patch_size, x:x+patch_size]

            # Check if more than 50% of the patch contains non-white pixels
            empty_percentage = np.sum(patch_greyscale > 150) / float(patch_greyscale.size)

            # Check if all pixels in the patch are in the tumorous area
            if np.all(mask_patch == 255) and empty_percentage < empty_threshold:

                # Save the patch from the Whole Slide Image
                patch_img = Image.fromarray(patch)
                patch_img.save(os.path.join(output_folder, f'patch_{x}_{y}.png'))

    # Close the Whole Slide Image
    slide.close()

if __name__ == "__main__":
    svs_file = "/images/PublicDatasets/NSCLC/TCGA_Renal/TCGA_SVS_Collect_COMBINED/TCGA-UW-A7GI-01Z-00-DX1.4A1709FC-F8F3-4509-A011-10CBF1B53DA9.svs"
    mask_file = "/images/PublicDatasets/NSCLC/TCGA_Renal/RCC_Annotated_Patches/complete_region_annotation/0a709187-bb10-4146-80f4-f46894d12a4d.png"
    output_directory = "/images/PublicDatasets/NSCLC/TCGA_Renal/RCC_Annotated_Patches/extracted_patches_annotated"
    patch_size = 512  # Adjust the patch size as needed

    extract_tumorous_patches(svs_file, mask_file, output_directory, patch_size)