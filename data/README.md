# Data

This project supports two data modes:

1. **Real product data**: provide a CSV with columns `id`, `image_path`, `title`, `description`, and `label`.
2. **Synthetic demo data**: generated locally with `python -m src.make_demo_data`.

The generated demo data is intentionally small and synthetic. It is useful for checking the pipeline, UI, and tests, but it is not evidence of real-world model quality.

Images referenced by the CSV may be absolute paths or paths relative to the CSV file.
