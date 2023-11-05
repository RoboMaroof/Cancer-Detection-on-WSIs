import os
import csv
import openslide

# Define the directories to search for .svs files
directories = ["/images/PublicDatasets/NSCLC/TCGA_Renal/TCGA-KIRC_SVS_Collect", "/images/PublicDatasets/NSCLC/TCGA_Renal/TCGA-KICH_SVS_Collect", "/images/PublicDatasets/NSCLC/TCGA_Renal/TCGA-KIRP_SVS_Collect"]

# Create or open a CSV file for writing
csv_file = open("metadata.csv", "w", newline="")
csv_writer = csv.writer(csv_file)

# Write the header row to the CSV file
csv_writer.writerow(["Slide_id", "Label", "Magnification"])

# Function to extract metadata from a single .svs file
def extract_metadata(slide_path, label):
    try:
        slide = openslide.OpenSlide(slide_path)
        magnification = slide.properties.get("aperio.AppMag")
        slide_id = os.path.splitext(os.path.basename(slide_path))[0]
        csv_writer.writerow([slide_id, label, magnification])
        slide.close()
    except Exception as e:
        print(f"Error processing {slide_path}: {str(e)}")

# Loop through the specified directories and process .svs files
for directory in directories:
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".svs"):
                slide_path = os.path.join(root, file)
                extract_metadata(slide_path, os.path.basename(root))

# Close the CSV file
csv_file.close()
