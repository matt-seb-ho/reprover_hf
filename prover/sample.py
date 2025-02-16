"""
Sampling proof search trees to collect negative data
"""
import os
import uuid
import json
import pickle
import hashlib
import argparse
from loguru import logger

from dotenv import load_dotenv
load_dotenv("/mnt/hdd/msho/gfn_ntp/src/.env")
os.environ["CACHE_DIR"] = "/mnt/hdd/msho/gfn_ntp/.cache/lean_dojo"
print("github access token passed in:", "GITHUB_ACCESS_TOKEN" in os.environ)

from lean_dojo import Theorem
from typing import List, Tuple, Optional
from lean_dojo import LeanGitRepo, Theorem, Pos, is_available_in_cache

from common import set_logger
from prover.proof_search import Status, DistributedProver, SearchResult
from prover.evaluate import _get_theorems
from prover.search_tree import Edge, InternalNode

def sample_trees(
    data_path: str,
    exp_id: Optional[str] = None,
    split: str = "val",
    file_path: Optional[str] = None,
    full_name: Optional[str] = None,
    name_filter: Optional[str] = None,
    num_theorems: Optional[int] = None,
    ckpt_path: Optional[str] = None,
    indexed_corpus_path: Optional[str] = None,
    tactic: Optional[str] = None,
    module: Optional[str] = None,
    num_sampled_tactics: int = 64,
    timeout: int = 600,
    num_cpus: int = 1,
    with_gpus: bool = False,
    verbose: bool = False,
    hf_generator_id: Optional[str] = None,
    hf_retrieval_id: Optional[str] = None,
    output_tree_file: Optional[str] = None,
) -> Tuple[float, list[dict]]:
    set_logger(verbose)

    repo, theorems, positions = _get_theorems(
        data_path, split, file_path, full_name, name_filter, num_theorems
    )

    # Search for proofs using multiple concurrent provers.
    prover = DistributedProver(
        ckpt_path,
        indexed_corpus_path,
        tactic,
        module,
        num_cpus,
        with_gpus=with_gpus,
        timeout=timeout,
        num_sampled_tactics=num_sampled_tactics,
        debug=verbose,
        hf_generator_id=hf_generator_id,
        hf_retriever_id=hf_retrieval_id,
    )
    results, trees = prover.search_unordered_and_return_trees(repo, theorems, positions)

    if output_tree_file:
        tree_data = {res.theorem.full_name: tree for res, tree in zip(results, trees) if res is not None}
        with open(output_tree_file, 'w') as f:
            json.dump(tree_data, f)
            logger.info(f"Sampled trees written out to: {output_tree_file}")


    # Calculate the result statistics.
    num_proved = num_failed = num_discarded = 0
    for r in results:
        if r is None:
            num_discarded += 1
        elif r.status == Status.PROVED:
            num_proved += 1
        else:
            num_failed += 1

    logger.info(
        f"Evaluation done! {num_proved} theorems proved, {num_failed} theorems failed, {num_discarded} non-theorems discarded"
    )

    if num_proved + num_failed == 0:
        pass_1 = float("nan")
    else:
        pass_1 = num_proved / (num_proved + num_failed)

    # Save the results.
    if exp_id is None:
        exp_id = str(uuid.uuid4())
    pickle_path = f"{exp_id}_results.pickle"
    pickle.dump(results, open(pickle_path, "wb"))
    logger.info(f"Results saved to {pickle_path}")

    return pass_1, trees


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script for evaluating the prover on theorems extracted by LeanDojo."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the data extracted by LeanDojo (e.g., data/leandojo_benchmark/random).",
    )
    parser.add_argument("--exp-id", type=str, help="Experiment ID used for logging.")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="val",
    )
    # `file_path`, `full_name`, `name_filter`, and `num_theorems` can be used to filter theorems.
    parser.add_argument("--file-path", type=str)
    parser.add_argument("--full-name", type=str)
    parser.add_argument("--name-filter", type=str)
    parser.add_argument("--num-theorems", type=int)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="Checkpoint of the tactic generator.",
    )
    parser.add_argument(
        "--hf_gen_id",
        type=str,
        help="hf repo/id of the tactic generator.",
    )
    parser.add_argument(
        "--hf_ret_id",
        type=str,
        help="hf repo/id of the retriever.",
    )
    parser.add_argument(
        "--indexed-corpus-path",
        type=str,
        help="Path to a pickled indexed corpus. Not required for models w/o retrieval.",
    )
    parser.add_argument("--tactic", type=str, help="The tactic to evaluate.")
    parser.add_argument("--module", type=str, help="The module to import the tactic.")
    parser.add_argument(
        "--num-sampled-tactics",
        type=int,
        default=64,
        help="Number of tactics to sample at each node during proof search.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Maximum number of seconds the proof search can take.",
    )
    parser.add_argument(
        "--num-cpus", type=int, default=1, help="The number of concurrent provers."
    )
    parser.add_argument(
        "--with-gpus", action="store_true", help="Use GPUs for proof search."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Set the logging level to DEBUG."
    )
    parser.add_argument(
        "--output_tree_file",
        type=str,
        help="json file to write sampled trees out to",
    )
    parser.add_argument(
        "--lean_dojo_cache_path",
        type=str,
        help="lean dojo downloads to a cache dir (defaults to Path.home())",
    )
    args = parser.parse_args()

    assert args.ckpt_path or args.tactic or args.hf_gen_id

    logger.info(f"PID: {os.getpid()}")
    logger.info(args)

    # # supplying github token increases my rate limits
    # if args.lean_dojo_cache_path:
    #     os.environ["CACHE_DIR"] = args.lean_dojo_cache_path
    #     # assert False

    pass_1, trees = sample_trees(
        args.data_path,
        args.exp_id,
        args.split,
        args.file_path,
        args.full_name,
        args.name_filter,
        args.num_theorems,
        args.ckpt_path,
        args.indexed_corpus_path,
        args.tactic,
        args.module,
        args.num_sampled_tactics,
        args.timeout,
        args.num_cpus,
        args.with_gpus,
        args.verbose,
        hf_generator_id=args.hf_gen_id,
        hf_retrieval_id=args.hf_ret_id,
        output_tree_file=args.output_tree_file
    )

    logger.info(f"Pass@1: {pass_1}")
    logger.info(f"Num trees: {len(trees)}")


if __name__ == "__main__":
    main()