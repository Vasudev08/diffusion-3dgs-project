"""
PyCOLMAP integration for automated preprocessing.
"""

import shutil
from pathlib import Path
from typing import Any

from pycolmap import (
    Device,
    ExhaustiveMatchingOptions,
    IncrementalPipelineOptions,
    Reconstruction,
    SiftExtractionOptions,
    SiftMatchingOptions,
)


class COLMAPProcessor:
    """PyCOLMAP processor for automated reconstruction."""

    def __init__(self, quality: str = "high", device: str = "cpu"):
        self.quality: str = quality
        self.device: str = device
        self.quality_settings: dict[str, Any] = self._get_quality_settings()

    def _get_quality_settings(self) -> dict[str, Any]:
        """Get COLMAP settings based on quality level."""
        settings = {
            "high": {
                "feature_extraction": {
                    "max_image_size": 3200,
                    "max_num_features": 8192,
                    "first_octave": -1,
                    "num_octaves": 4,
                    "octave_resolution": 3,
                    "peak_threshold": 0.0067,
                    "edge_threshold": 10,
                    "max_num_orientations": 8,
                    "upright": False,
                },
                "feature_matching": {
                    "max_ratio": 0.8,
                    "max_distance": 0.7,
                    "cross_check": True,
                    "max_error": 4.0,
                    "max_num_trials": 10000,
                    "confidence": 0.999,
                    "max_num_matches": 32768,
                },
                "mapper": {
                    "min_num_matches": 15,
                    "max_num_models": 50,
                    "max_model_overlap": 20,
                    "min_model_size": 10,
                    "max_reproj_error": 4.0,
                    "max_extra_error": 2.0,
                    "ba_refine_focal_length": True,
                    "ba_refine_principal_point": False,
                    "ba_refine_extra_params": True,
                    "min_focal_length_ratio": 0.1,
                    "max_focal_length_ratio": 10.0,
                    "max_extra_param": 1.0,
                },
            },
            "medium": {
                "feature_extraction": {
                    "max_image_size": 1600,
                    "max_num_features": 4096,
                },
                "feature_matching": {
                    "max_ratio": 0.8,
                    "max_distance": 0.7,
                },
                "mapper": {
                    "min_num_matches": 10,
                    "max_reproj_error": 6.0,
                },
            },
            "low": {
                "feature_extraction": {
                    "max_image_size": 800,
                    "max_num_features": 2048,
                },
                "feature_matching": {
                    "max_ratio": 0.8,
                    "max_distance": 0.7,
                },
                "mapper": {
                    "min_num_matches": 8,
                    "max_reproj_error": 8.0,
                },
            },
        }

        return settings.get(self.quality, settings["medium"])

    def process_images(
        self, image_dir: Path, output_dir: Path, database_path: Path | None = None
    ) -> dict[str, Any]:
        """
        Process images with COLMAP reconstruction.

        Args:
            image_dir: Directory containing input images
            output_dir: Directory for COLMAP output
            database_path: Optional path to existing database

        Returns:
            Dictionary with processing results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        database_dir = output_dir / "database"
        features_dir = output_dir / "features"
        matches_dir = output_dir / "matches"
        sparse_dir = output_dir / "sparse"

        for dir_path in [database_dir, features_dir, matches_dir, sparse_dir]:
            dir_path.mkdir(exist_ok=True)

        # Step 1: Feature extraction
        print("Extracting features...")
        import pycolmap

        sift_options = SiftExtractionOptions()
        # Apply quality settings to sift options
        for key, value in self.quality_settings["feature_extraction"].items():
            if hasattr(sift_options, key):
                setattr(sift_options, key, value)

        pycolmap.extract_features(
            database_path=str(database_dir / "database.db"),
            image_path=str(image_dir),
            sift_options=sift_options,
            device=Device.cuda if self.device == "cuda" else Device.cpu,
        )

        # Step 2: Feature matching
        print("Matching features...")
        sift_matching_options = SiftMatchingOptions()
        exhaustive_matching_options = ExhaustiveMatchingOptions()

        # Apply quality settings to matching options
        for key, value in self.quality_settings["feature_matching"].items():
            if hasattr(sift_matching_options, key):
                setattr(sift_matching_options, key, value)
            elif hasattr(exhaustive_matching_options, key):
                setattr(exhaustive_matching_options, key, value)

        pycolmap.match_exhaustive(
            database_path=str(database_dir / "database.db"),
            sift_options=sift_matching_options,
            matching_options=exhaustive_matching_options,
            device=Device.cuda if self.device == "cuda" else Device.cpu,
        )

        # Step 3: Reconstruction
        print("Running reconstruction...")
        pipeline_options = IncrementalPipelineOptions()

        # Apply quality settings to pipeline options
        for key, value in self.quality_settings["mapper"].items():
            if hasattr(pipeline_options, key):
                setattr(pipeline_options, key, value)

        reconstructions = pycolmap.incremental_mapping(
            database_path=str(database_dir / "database.db"),
            image_path=str(image_dir),
            output_path=str(sparse_dir),
            options=pipeline_options,
        )

        # Get the first (and typically only) reconstruction
        reconstruction = list(reconstructions.values())[0] if reconstructions else None

        if reconstruction is None:
            return {
                "success": False,
                "error": "Reconstruction failed - no valid reconstruction found",
                "output_dir": output_dir,
            }

        # Export to standard COLMAP format
        self._export_colmap_format(reconstruction, sparse_dir)

        return {
            "success": True,
            "reconstruction": reconstruction,
            "output_dir": output_dir,
            "sparse_dir": sparse_dir,
            "database_path": database_dir / "database.db",
        }

    def _export_colmap_format(self, reconstruction: Reconstruction, output_dir: Path):
        """Export reconstruction to standard COLMAP format."""
        # Export the entire reconstruction as binary format
        reconstruction.write_binary(str(output_dir))

    def validate_reconstruction(self, sparse_dir: Path) -> dict[str, Any]:
        """Validate the COLMAP reconstruction."""
        # Load the reconstruction
        import pycolmap

        reconstruction = pycolmap.Reconstruction()
        reconstruction.read_binary(str(sparse_dir))

        num_cameras = len(reconstruction.cameras)
        num_images = len(reconstruction.images)
        num_points = len(reconstruction.points3D)

        # Check reconstruction quality
        if num_images < 3:
            quality = "poor"
        elif num_images < 10:
            quality = "fair"
        elif num_images < 20:
            quality = "good"
        else:
            quality = "excellent"

        return {
            "valid": True,
            "num_cameras": num_cameras,
            "num_images": num_images,
            "num_points": num_points,
            "quality": quality,
        }

    def create_gsplat_dataset(
        self, sparse_dir: Path, images_dir: Path, output_dir: Path
    ) -> dict[str, Any]:
        """Create a dataset compatible with gsplat training."""
        # Copy images to output directory
        output_images_dir = output_dir / "images"
        output_images_dir.mkdir(parents=True, exist_ok=True)

        for img_file in images_dir.glob("*"):
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                _ = shutil.copy2(img_file, output_images_dir / img_file.name)

        # Copy sparse reconstruction
        output_sparse_dir = output_dir / "sparse" / "0"
        output_sparse_dir.mkdir(parents=True, exist_ok=True)

        for file in sparse_dir.glob("*.bin"):
            _ = shutil.copy2(file, output_sparse_dir / file.name)

        return {
            "success": True,
            "output_dir": output_dir,
            "images_dir": output_images_dir,
            "sparse_dir": output_sparse_dir,
        }
