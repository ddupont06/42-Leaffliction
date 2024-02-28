import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s |:| %(levelname)s |:| %(message)s",
    datefmt="%y/%m/%Y %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)
