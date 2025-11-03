Processed datasets live here.

Each object gets its own folder created by the pipeline or `ns-process-data`:

- datasets/<object>/
  - images/              # copied/downscaled images used by Nerfstudio
  - transforms.json      # camera poses + intrinsics (Nerfstudio format)

Do not place your raw photos here manually; put them under `datasets_raw/<object>/`.
