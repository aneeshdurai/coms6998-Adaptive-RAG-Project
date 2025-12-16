# from datasets import load_dataset
# from pathlib import Path
# import pandas as pd
# from workspace.utils.logging_utils import get_logger

# logger = get_logger("data.download_dataset")

# def download_financial_qa_10k(
#     hf_id: str,
#     out_dir: str,
#     splits=("train", "test"),
# ) -> None:
#     outp = Path(out_dir)
#     outp.mkdir(parents=True, exist_ok=True)

#     for split in splits:
#         logger.info(f"Downloading {hf_id} split={split}")
#         ds = load_dataset(hf_id, split=split)
#         df = ds.to_pandas()
#         out_file = outp / f"{hf_id.replace('/', '__')}_{split}.parquet"
#         df.to_parquet(out_file, index=False)
#         logger.info(f"Saved {split} -> {out_file}")

from datasets import load_dataset, get_dataset_split_names
from pathlib import Path
from workspace.utils.logging_utils import get_logger

logger = get_logger("data.download_dataset")

def download_financial_qa_10k(
    hf_id: str,
    out_dir: str,
    splits=("train", "test"),
) -> None:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    # Detect available splits
    try:
        available = get_dataset_split_names(hf_id)
    except Exception:
        # Fallback: load dataset dict and inspect keys
        ds_dict = load_dataset(hf_id)
        available = list(ds_dict.keys())

    logger.info(f"Available splits for {hf_id}: {available}")

    # Only download splits that actually exist
    for split in splits:
        if split not in available:
            logger.warning(f"Split '{split}' not found. Skipping.")
            continue

        logger.info(f"Downloading {hf_id} split={split}")
        ds = load_dataset(hf_id, split=split)
        df = ds.to_pandas()
        out_file = outp / f"{hf_id.replace('/', '__')}_{split}.parquet"
        df.to_parquet(out_file, index=False)
        logger.info(f"Saved {split} -> {out_file}")
