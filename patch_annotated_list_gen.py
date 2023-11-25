import os
import csv

def read_slide_list(file_path):
    slide_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip().split('\t')
            slide_dict[line[1]] = line[2]
    return slide_dict

def read_labeled_wsi_list(file_path):
    labeled_wsi_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip().split(',')
            labeled_wsi_dict[line[0][:23]] = line[1]
    return labeled_wsi_dict

def main():
    input_directory = '/images/PublicDatasets/NSCLC/TCGA_Renal/RCC_Annotated_Patches/complete_region_annotation'
    output_csv = '/images/PublicDatasets/NSCLC/TCGA_Renal/RCC_Annotated_Patches/patch_annotated_list.csv'
    slide_list_file = '/images/PublicDatasets/NSCLC/TCGA_Renal/RCC_Annotated_Patches/slide_list.txt'
    labeled_wsi_file = '/images/PublicDatasets/NSCLC/TCGA_Renal/RCC_Annotated_Patches/labeled_WSI_list.txt'

    # Task 1: Write the names of all PNG files in the input directory to the first column of a new CSV file
    png_files = [file.split('.')[0] for file in os.listdir(input_directory) if file.endswith('.png')]
    with open(output_csv, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['PNG_File_Name'])

        for png_file in png_files:
            csv_writer.writerow([png_file])

    # Task 2: Find and write corresponding names from slide_list.txt
    slide_dict = read_slide_list(slide_list_file)
    with open(output_csv, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip the header

        rows = [row for row in csv_reader]

    with open(output_csv, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['PNG_File_Name', 'Slide_Name'])

        for row in rows:
            png_file_name = row[0]
            if png_file_name in slide_dict:
                csv_writer.writerow([png_file_name, slide_dict[png_file_name]])

    # Task 3: Find and add values from labeled_WSI_list.txt
    labeled_wsi_dict = read_labeled_wsi_list(labeled_wsi_file)
    with open(output_csv, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip the header

        rows = [row for row in csv_reader]

    with open(output_csv, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['PNG_File_Name', 'Slide_Name', 'Label'])

        for row in rows:
            slide_name = row[1]
            if slide_name[:23] in labeled_wsi_dict:
                csv_writer.writerow([row[0], row[1], labeled_wsi_dict[slide_name[:23]]])

if __name__ == "__main__":
    main()
