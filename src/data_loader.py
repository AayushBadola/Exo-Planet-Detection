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

    pass