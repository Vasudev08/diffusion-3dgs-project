import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agentic_image2dataset.utils.download import download_weights


class TestWeightDownload(unittest.TestCase):
    @patch("agentic_image2dataset.utils.download.HfApi")
    @patch("agentic_image2dataset.utils.download.hf_hub_download")
    @patch("agentic_image2dataset.utils.download.shutil.copy2")
    def test_download_weights(self, mock_copy, mock_download, mock_hf_api):
        # Setup mocks
        mock_api_instance = MagicMock()
        mock_hf_api.return_value = mock_api_instance

        # Mock file list for AdcSR
        mock_api_instance.list_repo_files.return_value = [
            "README.md",
            "weight/",
            "weight/pretrained/halfDecoder.ckpt",
            "weight/net_params_200.pkl",
        ]

        # Mock download return value
        mock_download.return_value = "/tmp/cache/file"

        # Create a temporary directory for weights
        with patch("pathlib.Path.exists") as mock_exists:
            # Simulate that files do NOT exist locally
            mock_exists.return_value = False

            # Run download
            weights_dir = Path("test_weights")
            download_weights(weights_dir)

            # Verify AdcSR downloads
            # Should download weight/pretrained/halfDecoder.ckpt and weight/net_params_200.pkl
            # And HYPIR_sd2.pth

            # Check hf_hub_download calls
            # We expect 3 calls: 2 for AdcSR, 1 for HYPIR
            self.assertEqual(mock_download.call_count, 3)

            # Verify AdcSR calls
            mock_download.assert_any_call(
                repo_id="Guaishou74851/AdcSR",
                filename="weight/pretrained/halfDecoder.ckpt",
            )
            mock_download.assert_any_call(
                repo_id="Guaishou74851/AdcSR", filename="weight/net_params_200.pkl"
            )

            # Verify HYPIR call
            mock_download.assert_any_call(
                repo_id="lxq007/HYPIR", filename="HYPIR_sd2.pth"
            )

            # Verify copy calls
            self.assertEqual(mock_copy.call_count, 3)

    @patch("agentic_image2dataset.utils.download.HfApi")
    @patch("agentic_image2dataset.utils.download.hf_hub_download")
    def test_skip_existing_weights(self, mock_download, mock_hf_api):
        # Setup mocks
        mock_api_instance = MagicMock()
        mock_hf_api.return_value = mock_api_instance
        mock_api_instance.list_repo_files.return_value = ["weight/file1.txt"]

        with patch("pathlib.Path.exists") as mock_exists:
            # Simulate that files DO exist
            mock_exists.return_value = True

            download_weights(Path("test_weights"))

            # Should NOT download anything
            mock_download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
