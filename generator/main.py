"""Script for training the tactic generator."""

import os
from loguru import logger
from pytorch_lightning.cli import LightningCLI

from generator.datamodule import GeneratorDataModule
from generator.model import RetrievalAugmentedGenerator

generator_link_arguments = {
    "model.model_name": "data.model_name",
    "data.max_inp_seq_len": "model.max_inp_seq_len",
    "data.max_oup_seq_len": "model.max_oup_seq_len",
}


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        for k, v in generator_link_arguments.items():
            parser.link_arguments(k, v)


def main() -> None:
    logger.info(f"PID: {os.getpid()}")
    cli = CLI(RetrievalAugmentedGenerator, GeneratorDataModule)
    print("Configuration: \n", cli.config)


if __name__ == "__main__":
    main()
