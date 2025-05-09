import os
from typing import Any

import torch
from tqdm import tqdm

import sae_bench.evals.absorption.main as absorption
import sae_bench.evals.autointerp.main as autointerp
import sae_bench.evals.core.main as core
import sae_bench.evals.ravel.main as ravel
import sae_bench.evals.scr_and_tpp.main as scr_and_tpp
import sae_bench.evals.sparse_probing.main as sparse_probing
import sae_bench.evals.unlearning.main as unlearning
import sae_bench.sae_bench_utils.general_utils as general_utils

RANDOM_SEED = 42

MODEL_CONFIGS = {
    "pythia-70m-deduped": {
        "batch_size": 512,
        "dtype": "float32",
        "layers": [3, 4],
        "d_model": 512,
    },
    "gemma-2-2b": {
        "batch_size": 32,
        "dtype": "bfloat16",
        "layers": [12],
        "d_model": 2304,
    },
    "pythia-160m": {
        "batch_size": 512,
        "dtype": "float32",
        "layers": [8],
        "d_model": 1024,
    },
}

output_folders = {
    "absorption": "eval_results/absorption",
    "autointerp": "eval_results/autointerp",
    "core": "eval_results/core",
    "scr": "eval_results/scr",
    "tpp": "eval_results/tpp",
    "sparse_probing": "eval_results/sparse_probing",
    "unlearning": "eval_results/unlearning",
    "ravel": "eval_results/ravel",
}


def run_evals(
    model_name: str,
    selected_saes: list[tuple[str, Any]],
    llm_batch_size: int,
    llm_dtype: str,
    device: str,
    eval_types: list[str],
    api_key: str | None = None,
    force_rerun: bool = False,
    save_activations: bool = False,
):
    """Run selected evaluations for the given model and SAEs."""

    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}")

    # Mapping of eval types to their functions and output paths
    eval_runners = {
        "absorption": (
            lambda: absorption.run_eval(
                absorption.AbsorptionEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                "eval_results/absorption",
                force_rerun,
            )
        ),
        "autointerp": (
            lambda: autointerp.run_eval(
                autointerp.AutoInterpEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                api_key,  # type: ignore
                "eval_results/autointerp",
                force_rerun,
            )
        ),
        # TODO: Do a better job of setting num_batches and batch size
        "core": (
            lambda: core.multiple_evals(
                selected_saes=selected_saes,
                n_eval_reconstruction_batches=3200,
                n_eval_sparsity_variance_batches=2000,
                eval_batch_size_prompts=16,
                compute_featurewise_density_statistics=True,
                compute_featurewise_weight_based_metrics=True,
                exclude_special_tokens_from_reconstruction=True,
                dataset="Skylion007/openwebtext",
                context_size=128,
                output_folder="eval_results/core",
                verbose=True,
                dtype=llm_dtype,
                device=device,
            )
        ),
        "ravel": (
            lambda: ravel.run_eval(
                ravel.RAVELEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    llm_batch_size=llm_batch_size // 4,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                "eval_results/ravel",
                force_rerun,
            )
        ),
        "scr": (
            lambda: scr_and_tpp.run_eval(
                scr_and_tpp.ScrAndTppEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    perform_scr=True,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                "eval_results",  # We add scr or tpp depending on perform_scr
                force_rerun,
                clean_up_activations=True,
                save_activations=save_activations,
            )
        ),
        "tpp": (
            lambda: scr_and_tpp.run_eval(
                scr_and_tpp.ScrAndTppEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    perform_scr=False,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                "eval_results",  # We add scr or tpp depending on perform_scr
                force_rerun,
                clean_up_activations=True,
                save_activations=save_activations,
            )
        ),
        "sparse_probing": (
            lambda: sparse_probing.run_eval(
                sparse_probing.SparseProbingEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                "eval_results/sparse_probing",
                force_rerun,
                clean_up_activations=True,
                save_activations=save_activations,
            )
        ),
        "unlearning": (
            lambda: unlearning.run_eval(
                unlearning.UnlearningEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    llm_dtype=llm_dtype,
                    llm_batch_size=llm_batch_size // 8,
                ),
                selected_saes,
                device,
                "eval_results/unlearning",
                force_rerun,
            )
        ),
    }

    # Run selected evaluations
    for eval_type in tqdm(eval_types, desc="Evaluations"):
        if eval_type == "autointerp" and api_key is None:
            print("Skipping autointerp evaluation due to missing API key")
            continue
        # if eval_type == "unlearning":
            # if model_name != "gemma-2-2b":
            #     print("Skipping unlearning evaluation for non-GEMMA model")
            #     continue
            # print("Skipping, need to clean up unlearning interface")
            # continue # TODO:
            # if not os.path.exists(
            #     "./sae_bench/evals/unlearning/data/bio-forget-corpus.jsonl"
            # ):
            #     print(
            #         "Skipping unlearning evaluation due to missing bio-forget-corpus.jsonl"
            #     )
            #     continue

        print(f"\n\n\nRunning {eval_type} evaluation\n\n\n")

        if eval_type in eval_runners:
            os.makedirs(output_folders[eval_type], exist_ok=True)
            eval_runners[eval_type]()


if __name__ == "__main__":
    import sae_bench.custom_saes.identity_sae as identity_sae

    device = general_utils.setup_environment()

    model_name = "pythia-70m-deduped"
    model_name = "gemma-2-2b"
    d_model = MODEL_CONFIGS[model_name]["d_model"]
    llm_batch_size = MODEL_CONFIGS[model_name]["batch_size"]
    llm_dtype = MODEL_CONFIGS[model_name]["dtype"]

    # Note: Unlearning is not recommended for models with < 2B parameters and we recommend an instruct tuned model
    # Unlearning will also require requesting permission for the WMDP dataset (see unlearning/README.md)
    # Absorption not recommended for models < 2B parameters

    # Select your eval types here.
    eval_types = [
        "absorption",
        "autointerp",
        "core",
        "ravel",
        "scr",
        "tpp",
        "sparse_probing",
        "unlearning",
    ]

    if "autointerp" in eval_types:
        try:
            with open("openai_api_key.txt") as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            raise Exception("Please create openai_api_key.txt with your API key")
    else:
        api_key = None

    if "unlearning" in eval_types:
        if not os.path.exists(
            "./sae_bench/evals/unlearning/data/bio-forget-corpus.jsonl"
        ):
            raise Exception(
                "Please download bio-forget-corpus.jsonl for unlearning evaluation"
            )

    # If evaluating multiple SAEs on the same layer, set save_activations to True
    # This will require at least 100GB of disk space
    save_activations = False

    for hook_layer in MODEL_CONFIGS[model_name]["layers"]:
        sae = identity_sae.IdentitySAE(
            d_model,
            model_name,
            hook_layer,
            device=torch.device(device),
            dtype=general_utils.str_to_dtype(llm_dtype),
        )  # type: ignore
        selected_saes = [(f"{model_name}_layer_{hook_layer}_identity_sae", sae)]

        # This will evaluate PCA SAEs
        # sae = pca_sae.PCASAE(
        #     d_model,
        #     model_name,
        #     hook_layer,
        #     device=torch.device(device),
        #     dtype=general_utils.str_to_dtype(llm_dtype),
        # )
        # filename = f"pca_gemma-2-2b_blocks.{hook_layer}.hook_resid_post.pt"
        # sae.load_from_file(filename)
        # selected_saes = [(f"{model_name}_layer_{hook_layer}_pca_sae", sae)]

        for sae_name, sae in selected_saes:
            sae = sae.to(dtype=general_utils.str_to_dtype(llm_dtype))
            sae.cfg.dtype = llm_dtype

        run_evals(
            model_name,
            selected_saes,
            llm_batch_size,
            llm_dtype,
            device,
            eval_types=eval_types,
            api_key=api_key,
            force_rerun=False,
            save_activations=False,
        )
