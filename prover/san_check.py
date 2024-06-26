from common import prepare_environment_for_lean_dojo
prepare_environment_for_lean_dojo("config.yaml") # need to run from repo root

import torch
from generator.model import RetrievalAugmentedGenerator

if __name__ == "__main__":
    hf_generator_id = "kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small"
    hf_retriever_id = "kaiyuy/leandojo-lean4-retriever-byt5-small"
    device = torch.device("cuda")

    tac_gen = RetrievalAugmentedGenerator.load_from_hf(
        hf_generator_id, 
        hf_retriever_id=hf_retriever_id,
        device=device,
    )

    print("Loaded, yipeee!")