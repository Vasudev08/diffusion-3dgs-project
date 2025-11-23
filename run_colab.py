"""
Simple wrapper to run the colab pipeline from a Colab notebook cell.
This avoids the IPython kernel error when running as a script.

Usage in Colab:
    import run_colab
    run_colab.run()
"""

def run():
    """Run the colab pipeline."""
    from colab_pipeline import main
    main()

if __name__ == "__main__":
    run()
