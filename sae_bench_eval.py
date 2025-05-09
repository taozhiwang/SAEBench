import os
import torch
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Optional, Union

import sae_bench.custom_saes.custom_sae_config as custom_sae_config
import sae_bench.custom_saes.relu_sae as relu_sae
import sae_bench.evals.core.main as core
import sae_bench.evals.sparse_probing.main as sparse_probing
import sae_bench.sae_bench_utils.general_utils as general_utils
import sae_bench.sae_bench_utils.graphing_utils as graphing_utils
from sae_bench.sae_bench_utils.sae_selection_utils import get_saes_from_regex
from sae_lens import SAE


class SAEEvaluator:
    """Class for evaluating Sparse Autoencoder (SAE) models"""
    
    def __init__(
        self, 
        model_name: str = "pythia-160m",
        gpu_id: str = "0",
        random_seed: int = 42,
        output_dir: str = "eval_results",
        torch_dtype: torch.dtype = torch.float32,
        llm_batch_size: int = 512,
        save_activations: bool = False
    ):
        """Initialize the SAE evaluator
        
        Args:
            model_name: Base language model name
            gpu_id: GPU ID to use
            random_seed: Random seed
            output_dir: Result output directory
            torch_dtype: Tensor data type
            llm_batch_size: Language model batch size
            save_activations: Whether to save activations for reuse by multiple SAEs
        """
        self.model_name = model_name
        self.random_seed = random_seed
        self.torch_dtype = torch_dtype
        self.llm_batch_size = llm_batch_size
        self.save_activations = save_activations
        
        # Set up GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        self.device = general_utils.setup_environment()
        self.str_dtype = torch_dtype.__str__().split(".")[-1]
        
        # Create output directories
        self.output_folders = {
            "absorption": f"{output_dir}/absorption",
            "autointerp": f"{output_dir}/autointerp",
            "core": f"{output_dir}/core",
            "scr": f"{output_dir}/scr",
            "tpp": f"{output_dir}/tpp",
            "sparse_probing": f"{output_dir}/sparse_probing",
            "unlearning": f"{output_dir}/unlearning",
            "ravel": f"{output_dir}/ravel",
        }
        
        for folder in self.output_folders.values():
            os.makedirs(folder, exist_ok=True)
        
        # Chart style configuration
        self.trainer_markers = {
            "standard": "o",
            "jumprelu": "X",
            "topk": "^",
            "p_anneal": "*",
            "gated": "d",
            "vanilla": "s",
        }
        
        self.trainer_colors = {
            "standard": "blue",
            "jumprelu": "orange",
            "topk": "green",
            "p_anneal": "red",
            "gated": "purple",
            "vanilla": "black",
        }
        
        self.selected_saes = []

    def load_sae_from_directory(self, sae_dir: str, custom_id: str = None) -> SAE:
        """Load SAE model from directory
        
        Args:
            sae_dir: SAE model directory path
            custom_id: Custom SAE ID
            
        Returns:
            Loaded SAE model
        """
        llm_dtype = self.str_dtype
        
        if custom_id is None:
            custom_id = os.path.basename(sae_dir)
            
        print(f"Loading SAE {custom_id} from {sae_dir}")
        
        sae = SAE.load_from_pretrained(sae_dir, device=self.device, dtype=llm_dtype)
        
        # Fill in configuration fields expected by SAE Bench
        sae.cfg.dtype = llm_dtype
        sae.cfg.architecture = "standard"  # Use the architecture you trained with
        sae.cfg.training_tokens = 512_000_000  # Total tokens you trained on
        
        # Add these fields if not present in cfg
        if not hasattr(sae.cfg, "hook_layer"):
            sae.cfg.hook_layer = 8  # Modify to the correct layer
        
        if not hasattr(sae.cfg, "hook_name"):
            sae.cfg.hook_name = f"blocks.{sae.cfg.hook_layer}.hook_mlp_out"
            
        return sae
            
    def add_sae(self, sae_id: str, sae: SAE) -> None:
        """Add SAE to evaluation list
        
        Args:
            sae_id: SAE identifier
            sae: SAE object
        """
        self.selected_saes.append((sae_id, sae))
    
    def add_saes_from_config(self, sae_config: Dict[str, Dict[str, str]]) -> None:
        """Add multiple SAEs from configuration dictionary
        
        Args:
            sae_config: SAE configuration dictionary {sae_id: {"sae_dir": dir_path, "custom_id": id}}
        """
        for key in sae_config:
            sae_dir = sae_config[key]["sae_dir"]
            custom_id = sae_config[key]["custom_id"]
            
            sae = self.load_sae_from_directory(sae_dir)
            self.add_sae(custom_id, sae)
    
    def add_saes_from_regex(self, sae_regex_pattern: str, sae_block_pattern: str) -> None:
        """Add SAEs from SAE Lens using regex patterns
        
        Args:
            sae_regex_pattern: SAE regex pattern
            sae_block_pattern: Block pattern regex
        """
        regex_saes = get_saes_from_regex(sae_regex_pattern, sae_block_pattern)
        self.selected_saes.extend(regex_saes)
        print(f"Added {len(regex_saes)} SAEs from regex patterns")
    
    def run_core_evaluation(
        self,
        n_eval_reconstruction_batches: int = 200,
        n_eval_sparsity_variance_batches: int = 200,
        eval_batch_size_prompts: int = 32,
        dataset: str = "Skylion007/openwebtext",
        context_size: int = 128,
        force_rerun: bool = False,
        verbose: bool = True
    ) -> Dict:
        """Run core evaluation
        
        Args:
            n_eval_reconstruction_batches: Number of reconstruction evaluation batches
            n_eval_sparsity_variance_batches: Number of sparsity and variance evaluation batches
            eval_batch_size_prompts: Evaluation batch size
            dataset: Dataset to use
            context_size: Context size
            force_rerun: Whether to force re-running the evaluation
            verbose: Whether to display detailed output
            
        Returns:
            Evaluation results dictionary
        """
        print("\n\nRunning core evaluation\n\n")
        
        results = core.multiple_evals(
            selected_saes=self.selected_saes,
            n_eval_reconstruction_batches=n_eval_reconstruction_batches,
            n_eval_sparsity_variance_batches=n_eval_sparsity_variance_batches,
            eval_batch_size_prompts=eval_batch_size_prompts,
            compute_featurewise_density_statistics=True,
            compute_featurewise_weight_based_metrics=True,
            exclude_special_tokens_from_reconstruction=True,
            dataset=dataset,
            context_size=context_size,
            output_folder=self.output_folders["core"],
            verbose=verbose,
            dtype=self.str_dtype,
            force_rerun=force_rerun
        )
        
        return results
    
    def run_sparse_probing(
        self, 
        dataset_names: List[str] = None,
        force_rerun: bool = False,
        clean_up_activations: bool = True
    ) -> Dict:
        """Run sparse probing evaluation
        
        Args:
            dataset_names: List of dataset names
            force_rerun: Whether to force re-running the evaluation
            clean_up_activations: Whether to clean up activations
            
        Returns:
            Evaluation results dictionary
        """
        if dataset_names is None:
            dataset_names = ["LabHC/bias_in_bios_class_set1"]
        
        print("\n\nRunning sparse probing evaluation\n\n")
        
        results = sparse_probing.run_eval(
            sparse_probing.SparseProbingEvalConfig(
                model_name=self.model_name,
                random_seed=self.random_seed,
                llm_batch_size=self.llm_batch_size,
                llm_dtype=self.str_dtype,
                dataset_names=dataset_names,
            ),
            self.selected_saes,
            self.device,
            self.output_folders["sparse_probing"],
            force_rerun=force_rerun,
            clean_up_activations=clean_up_activations,
            save_activations=self.save_activations,
        )
        
        return results
    
    def run_all_evaluations(
        self, 
        eval_types: List[str] = None,
        force_rerun: bool = False
    ) -> Dict[str, Any]:
        """Run all selected evaluations
        
        Args:
            eval_types: List of evaluation types to run
            force_rerun: Whether to force re-running evaluations
            
        Returns:
            Dictionary of evaluation results {eval_type: results}
        """
        if eval_types is None:
            eval_types = ["core", "sparse_probing"]
        
        results = {}
        
        for eval_type in eval_types:
            print(f"\n\nRunning {eval_type} evaluation\n\n")
            
            if eval_type == "core":
                results[eval_type] = self.run_core_evaluation(force_rerun=force_rerun)
            elif eval_type == "sparse_probing":
                results[eval_type] = self.run_sparse_probing(force_rerun=force_rerun)
            # Add other evaluation types here
            
        return results
    
    def create_image_directory(self, image_path: str = "images") -> str:
        """Create image output directory
        
        Args:
            image_path: Image directory path
            
        Returns:
            Created directory path
        """
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        return image_path
    
    def plot_results(
        self, 
        eval_type: str = "sparse_probing", 
        k: int = 1,
        image_path: str = "images"
    ) -> None:
        """Plot evaluation results
        
        Args:
            eval_type: Evaluation type
            k: k value parameter (for TopK evaluations)
            image_path: Image output path
        """
        image_base_name = os.path.join(image_path, eval_type)
        
        eval_folders = [f"{os.path.dirname(self.output_folders[eval_type])}/{eval_type}"]
        core_folders = [f"{os.path.dirname(self.output_folders['core'])}/core"]
        
        eval_filenames = graphing_utils.find_eval_results_files(eval_folders)
        core_filenames = graphing_utils.find_eval_results_files(core_folders)
        
        graphing_utils.plot_results(
            eval_filenames,
            core_filenames,
            eval_type,
            image_base_name,
            k,
            trainer_markers=self.trainer_markers,
            trainer_colors=self.trainer_colors,
        )
        
        print(f"Plots saved to {image_path}")
    
    def print_result_summary(self, eval_type: str = "sparse_probing", k: int = 1) -> None:
        """Print evaluation results summary
        
        Args:
            eval_type: Evaluation type
            k: k value parameter (for TopK evaluations)
        """
        eval_folders = [f"{os.path.dirname(self.output_folders[eval_type])}/{eval_type}"]
        core_folders = [f"{os.path.dirname(self.output_folders['core'])}/core"]
        
        eval_filenames = graphing_utils.find_eval_results_files(eval_folders)
        
        if len(eval_filenames) == 0:
            print(f"No evaluation results found for {eval_type}")
            return
            
        # Display comparison of first and second SAE results
        if len(eval_filenames) >= 2:
            baseline_filepath = eval_filenames[0]
            custom_filepath = eval_filenames[1]
            
            with open(baseline_filepath) as f:
                baseline_results = json.load(f)
                
            with open(custom_filepath) as f:
                custom_results = json.load(f)
                
            print(f"Baseline SAE top {k} accuracy:", 
                  baseline_results["eval_result_metrics"]["sae"][f"sae_top_{k}_test_accuracy"])
            print(f"Custom SAE top {k} accuracy:", 
                  custom_results["eval_result_metrics"]["sae"][f"sae_top_{k}_test_accuracy"])
            
            if "llm" in baseline_results["eval_result_metrics"]:
                print(f"LLM residual stream top {k} accuracy:", 
                      baseline_results["eval_result_metrics"]["llm"][f"llm_top_{k}_test_accuracy"])


def main():
    """Main function example"""
    # Initialize evaluator
    evaluator = SAEEvaluator(
        model_name="pythia-160m",
        gpu_id="9",
        output_dir="eval_results"
    )
    
    # Configure SAEs to load
    sae_config = {
        "sae1": {
            "sae_dir": "/data/zixuan/phd/sae/results/5.1-grid_l1-1_newinit-True_seed-1_gated/final_512000000",
            "custom_id": "5.1-grid_l1-1_newinit-True_seed-1_gated"
        },
        "sae2": {
            "sae_dir": "/data/zixuan/phd/sae/results/5.1-grid_l1-1_newinit-False_seed-1_gated/final_512000000",
            "custom_id": "5.1-grid_l1-1_newinit-False_seed-1_gated"
        }
    }
    
    # Load SAE models
    evaluator.add_saes_from_config(sae_config)
    
    # Create image output directory
    image_path = evaluator.create_image_directory()
    
    # Run core evaluation
    evaluator.run_core_evaluation()
    
    # Run sparse probing evaluation
    evaluator.run_sparse_probing()
    
    # Plot results
    evaluator.plot_results(eval_type="sparse_probing", image_path=image_path)
    
    # Print results summary
    evaluator.print_result_summary(eval_type="sparse_probing")


if __name__ == "__main__":
    main()