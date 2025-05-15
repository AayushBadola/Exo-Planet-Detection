import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame | None:
    """
    Loads the raw data from the specified CSV file.
    """
    logger.info(f"Attempting to load data from: {file_path}")
    try:
        # Kepler data often has comments starting with #
        df = pd.read_csv(file_path, comment='#')
        if df.empty:
            logger.warning(f"Loaded an empty DataFrame from {file_path} with comment='#'. Trying without.")
            # Fallback if comment='#' leads to empty df but file might be valid otherwise
            df_fallback = pd.read_csv(file_path)
            if df_fallback.empty:
                logger.error(f"Still loaded an empty DataFrame from {file_path} even without comment='#' handling.")
                return None
            logger.info(f"Successfully loaded data using fallback (no comment handling). Shape: {df_fallback.shape}")
            return df_fallback

        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"No data: File is empty at {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    # This is for basic testing, not part of the main pipeline run
    # You'd need a config or hardcoded path here for standalone testing
    # For example:
    # cfg_test_path = "../../data/cumulative.csv" # Adjust path if running standalone
    # logging.basicConfig(level=logging.INFO)
    # data = load_data(cfg_test_path)
    # if data is not None:
    #     logger.info("First 5 rows of the loaded data:")
    #     logger.info(f"\n{data.head()}")
    #     logger.info(f"\nShape of the data: {data.shape}")
    pass