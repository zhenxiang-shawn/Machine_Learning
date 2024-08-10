"""
Rename dataset for hymenoptera dataset.
Convert format from {dir}/{filename} to {dir}/{filename} + {dir}/{label}

Author: Zhenxiang Jin (zhenxiang.shawn@zohomail.com)
"""
import os

datasets = ['train', 'val']

for dataset in datasets:
    dataset_abs_path = os.path.abspath(dataset)
    print("abs_path:", dataset_abs_path)
    dirs = os.listdir(dataset_abs_path)
    for directory in dirs:
        # rename each dir to class_image
        old_path = os.path.join(dataset_abs_path, directory)
        new_path = os.path.join(dataset_abs_path, f'{directory}_image')
        os.rename(old_path, new_path)
        print(f'going to rename {old_path} to \n {new_path}')
        # create label files
        label_dir = os.path.join(dataset_abs_path, f'{directory}_label')
        os.makedirs(label_dir, exist_ok=True)
        for filename in os.listdir(new_path):
            label_filename = filename.split('.')[0]
            with open(os.path.join(label_dir, label_filename + '.txt'),
                      'a+') as f:
                f.write(directory)
                print(f"Going to generate "
                      f"{os.path.join(label_dir, label_filename + '.txt')}" 
                      f"with content: {directory}")






# loop and generate label file

