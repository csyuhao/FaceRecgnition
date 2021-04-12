# Preparing Datasets

This folder is utilized to cropping images and detect the face in image

```Bash
cd preprocess
conda activate env-name
python align_images.py --input_dir=images --output_dir=output
python generate_dataset_list.py --dataset_path=output --dataset_list=list
```