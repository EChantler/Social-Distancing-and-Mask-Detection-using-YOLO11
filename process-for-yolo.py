import os
import shutil

# Define directories
raw_with_mask = os.listdir('raw/with_mask')
raw_without_mask = os.listdir('raw/without_mask')

# Helper function to split dataset
def split_dataset(raw_images, split_ratios=(0.8, 0.1, 0.1)):
    n = len(raw_images)
    train_split = int(n * split_ratios[0])
    val_split = int(n * (split_ratios[0] + split_ratios[1]))
    return raw_images[:train_split], raw_images[train_split:val_split], raw_images[val_split:]

# Split datasets
train, val, test = split_dataset(raw_with_mask)
train_no_mask, val_no_mask, test_no_mask = split_dataset(raw_without_mask)

# Create directories
for subset in ['train', 'val', 'test']:
    os.makedirs(f'images/{subset}', exist_ok=True)
    os.makedirs(f'labels/{subset}', exist_ok=True)

# Copy files and create labels
def process_images(images, class_id, subset):
    for img in images:
        shutil.copy(f'raw/with_mask/{img}' if class_id == 2 else f'raw/without_mask/{img}', f'images/{subset}/{img}')
        with open(f'labels/{subset}/{img[:-4]}.txt', 'w') as label_file:
            label_file.write(f"{class_id} 0.5 0.5 1 1\n")

process_images(train, 2, 'train')
process_images(val, 2, 'val')
process_images(test, 2, 'test')
process_images(train_no_mask, 1, 'train')
process_images(val_no_mask, 1, 'val')
process_images(test_no_mask, 1, 'test')
