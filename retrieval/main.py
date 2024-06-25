"""Script for training the premise retriever.
"""

import os
from loguru import logger
from pytorch_lightning.cli import LightningCLI

from retrieval.model import PremiseRetriever
from retrieval.datamodule import RetrievalDataModule


retriever_link_arguments = {
    "model.model_name": "data.model_name",
    "data.max_seq_len": "model.max_seq_len",
}


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        for k, v in retriever_link_arguments.items():
            parser.link_arguments(k, v)


def main() -> None:
    logger.info(f"PID: {os.getpid()}")
    cli = CLI(PremiseRetriever, RetrievalDataModule)
    logger.info("Configuration: \n", cli.config)


if __name__ == "__main__":
    main()
