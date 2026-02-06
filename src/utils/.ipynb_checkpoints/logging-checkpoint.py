import logging
from pathlib import Path


PARENT_PATH = Path(os.getenv("PARENT_DIR", "."))  # default fallback to current directory
logging.basicConfig(
    filename=PARENT_PATH / "app.log",
    filemode="a",  # append mode
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)