Put your original multi-view photos here, one folder per object.

Example:
- datasets_raw/mug_01/              # ← place your 17 photos in this folder
  - IMG_0001.jpg
  - IMG_0002.jpg
  - ... (total 17 images)

Then run:
  python single_image_pipeline/run_pipeline.py --images-dir datasets_raw/mug_01 --matcher exhaustive
  - make sure to run the above command inside the conda envirnoment.

Outputs will be created in:
- datasets/mug_01/      (processed images + transforms.json)
- results/mug_01/       (training runs, logs, exports)
