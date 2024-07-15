import h5py
import numpy as np

def convert_h5_to_npy(h5_path, output_dir):
    with h5py.File(h5_path, 'r') as h5_file:
        # Assuming the dataset structure has keys 'image' and 'label'
        image_data = h5_file['image'][()]
        label_data = h5_file['label'][()]

    # Save as .npy files
    np.save(output_dir + '/Image.npy', image_data)
    np.save(output_dir + '/Label.npy', label_data)
    print(f"Saved Image.npy and Label.npy to {output_dir}")

# Example usage
convert_h5_to_npy('/home/amishr17/aryan/new_attempt/preprocess/TBAD/ImageTBAD/TBAD-40.h5', '/home/amishr17/aryan/new_attempt/numpy_preprocess')
