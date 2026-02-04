import logging
import sys

# -------------------------------------------------
# Logger configuration
# -------------------------------------------------
logger = logging.getLogger("imdb_sentiment")
logger.setLevel(logging.INFO)          # Change to DEBUG for more detail

# Console handler (Streamlit’s server console)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Optional: file handler – uncomment if you want a log file
file_handler = logging.FileHandler("imdb_sentiment.log")
file_handler.setLevel(logging.INFO)

# Common log format
formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Attach handlers (avoid duplicate handlers on reload)
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)