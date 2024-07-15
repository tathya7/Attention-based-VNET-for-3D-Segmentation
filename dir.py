import os

def filter_file_paths(data_dir, output_file):
    valid_paths = []
    
    # Walk through the data_dir and collect valid .h5 file paths
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.h5'):
                path = os.path.join(root, file)
                if os.path.isfile(path):
                    valid_paths.append(path)
                else:
                    print(f"Skipping directory or non-existent file: {path}")
    
    # Write valid file paths to the output_file
    with open(output_file, 'w') as file:
        for path in valid_paths:
            file.write(path + '\n')
    
    print(f"Valid file paths have been written to {output_file}")

if __name__ == "__main__":
    data_dir = 'preprocess/TBAD'  # The base directory of your dataset
    output_file = 'datalist/AD/AD_0_filtered.txt'  # The output file to save the valid paths
    
    filter_file_paths(data_dir, output_file)
