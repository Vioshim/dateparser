from pathlib import Path
import urllib.request
import os
import logging

from .exceptions import FastTextModelNotFoundException
from .utils import dateparser_model_home, create_data_model_home


def fasttext_downloader(model_name):
    create_data_model_home()
    models = {
        "small": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",
        "large": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    }
    if model_name not in models:
        message = f"""dateparser-download: Couldn't find a model called \"{model_name}\". Supported models are: {", ".join(models.keys())}"""
        raise FastTextModelNotFoundException(message)

    models_directory_path = os.path.join(
        dateparser_model_home, f"{model_name}.bin"
    )

    if not Path(models_directory_path).is_file():
        model_url = models[model_name]
        logging.info(
            f'dateparser-download: Downloading model \"{model_name}\" from \"{model_url}\"...'
        )
        try:
            urllib.request.urlretrieve(model_url, models_directory_path)
        except urllib.error.HTTPError as e:
            raise Exception("dateparser-download: Fasttext model cannot be downloaded due to HTTP error") from e
    else:
        logging.info(
            f'dateparser-download: The model \"{model_name}\" is already downloaded'
        )
