Training outputs and exports live here.

Per object, the pipeline creates:
- results/<object>/
  - logs/               # CLI logs from ns-process, ns-train, ns-export
  - <run_subdir>/       # created by ns-train (config.yml, checkpoints/)
  - exports/            # exported Gaussian splats / PLYs
  - summary.json        # quick reference to key artifacts
