"""Script for indexing the corpus using the retriever.
"""
import torch
import pickle
import argparse
from loguru import logger

from common import IndexedCorpus
from retrieval.model import PremiseRetriever


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script for training the BM25 premise retriever."
    )
    # original line
    # parser.add_argument("--ckpt_path", type=str, required=True)
    # need to remove this dependency on ckpt_path
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--hf_model_id", type=str)
    parser.add_argument("--corpus-path", type=str, required=True)
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    logger.info(args)

    assert args.ckpt_path or args.hf_model_id, "Need to provide the retriever model somehow"

    if not torch.cuda.is_available():
        logger.warning("Indexing the corpus using CPU can be very slow.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    if args.ckpt_path is not None:
        model = PremiseRetriever.load(args.ckpt_path, device, freeze=True)
    else:
        # model = PremiseRetriever.load_from_hf(args.hf_model_id, device, freeze=True)
        # setting None device should make it use device_map="auto"
        model = PremiseRetriever.load_from_hf(args.hf_model_id, None, freeze=True)
    model.load_corpus(args.corpus_path)
    model.reindex_corpus(batch_size=args.batch_size)

    pickle.dump(
        IndexedCorpus(model.corpus, model.corpus_embeddings.cpu()),
        open(args.output_path, "wb"),
    )
    logger.info(f"Indexed corpus saved to {args.output_path}")


if __name__ == "__main__":
    main()
