import os
import torch
from typing import List, Dict, Any, Optional, Union

import sae_bench.custom_saes.run_all_evals_custom_saes as run_all_evals_custom_saes
import sae_bench.sae_bench_utils.general_utils as general_utils
from sae_bench.sae_bench_utils.sae_selection_utils import get_saes_from_regex
from sae_lens import SAE


class SAEEvaluator:
    """Class for evaluating Sparse Autoencoder (SAE) models"""
    
    def __init__(
        self, 
        model_name: str = "pythia-160m",
        gpu_id: str = "0",
        random_seed: int = 1,
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
        self.output_dir = output_dir
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
        
        # List to store selected SAEs
        self.selected_saes = []

    def normalize_sae(self, sae: SAE) -> SAE:
        """Normalize the decoder weights and adjust encoder weights accordingly.
        
        This normalization ensures that each feature vector (column of the decoder) has unit L2 norm.
        To preserve the SAE's function, the encoder weights are scaled inversely by the same factors.
        
        Args:
            sae: Sparse Autoencoder to normalize
            
        Returns:
            Normalized SAE with unchanged functionality
        """
        # Check if the SAE is using the gated architecture
        if not hasattr(sae, 'b_gate') or not hasattr(sae, 'b_mag'):
            raise ValueError("This normalization method is only applicable to SAEs with gated architecture. "
                            "The provided SAE does not have the required attributes (b_gate and b_mag).")

        decoder_weights = sae.W_dec.detach().clone()
        feature_norms = torch.norm(decoder_weights, p=2, dim=1, keepdim=True)

        # Avoid division by zero
        feature_norms = torch.clamp(feature_norms, min=1e-8)
        
        normalized_decoder = decoder_weights / feature_norms
        encoder_weights = sae.W_enc.detach().clone()
        scaled_encoder = encoder_weights * feature_norms.T
        
        b_gate = sae.b_gate.detach().clone()
        # Reshape to match expected dimensions for broadcasting
        feature_norms_for_bias = feature_norms.squeeze(1)  # Remove the keepdim dimension
        scaled_b_gate = b_gate * feature_norms_for_bias

        b_mag = sae.b_mag.detach().clone()
        scaled_b_mag = b_mag * feature_norms_for_bias

        # Update the SAE with normalized weights
        with torch.no_grad():
            sae.W_dec.copy_(normalized_decoder)
            sae.W_enc.copy_(scaled_encoder)
            sae.b_mag.copy_(scaled_b_mag)
            sae.b_gate.copy_(scaled_b_gate)
        return sae

    def add_saes_from_config(self, sae_config: Dict[str, Dict[str, str]], normalize: bool = False) -> None:
        """Add multiple SAEs from configuration dictionary
        
        Args:
            sae_config: SAE configuration dictionary {sae_id: {"sae_dir": dir_path, "custom_id": id}}
            normalize: Whether to normalize the decoder weights
        """
        for key in sae_config:
            sae_dir = sae_config[key]["sae_dir"]
            custom_id = sae_config[key]["custom_id"]
            
            sae = SAE.load_from_pretrained(sae_dir, device=self.device, dtype=self.str_dtype)
            
            # Optionally normalize the SAE
            if normalize:
                sae = self.normalize_sae(sae)
                
            self.selected_saes.append((custom_id, sae))
    
    def add_saes_from_regex(self, sae_regex_pattern: str, sae_block_pattern: str, normalize: bool = False) -> None:
        """Add SAEs from SAE Lens using regex patterns
        
        Args:
            sae_regex_pattern: SAE regex pattern
            sae_block_pattern: Block pattern regex
            normalize: Whether to normalize the decoder weights
        """
        regex_saes = get_saes_from_regex(sae_regex_pattern, sae_block_pattern)
        
        if normalize:
            normalized_saes = []
            for custom_id, sae in regex_saes:
                sae = self.normalize_sae(sae)
                normalized_saes.append((custom_id, sae))
            self.selected_saes.extend(normalized_saes)
        else:
            self.selected_saes.extend(regex_saes)
            
        print(f"Added {len(regex_saes)} SAEs from regex patterns")
    
    def run_evaluations(
        self, 
        eval_types: List[str] = None,
        force_rerun: bool = False,
        api_key: str = None
    ) -> Dict[str, Any]:
        """Run selected evaluations using run_all_evals_custom_saes
        
        Args:
            eval_types: List of evaluation types to run
            force_rerun: Whether to force re-running evaluations
            api_key: API key for certain evaluations
            
        Returns:
            Dictionary of evaluation results {eval_type: results}
        """
        if eval_types is None:
            eval_types = ["core", "sparse_probing"]
        
        print(f"\n\nRunning evaluations: {', '.join(eval_types)}\n\n")
        
        return run_all_evals_custom_saes.run_evals(
            model_name=self.model_name,
            selected_saes=self.selected_saes,
            llm_batch_size=self.llm_batch_size,
            llm_dtype=self.str_dtype,
            device=self.device,
            eval_types=eval_types,
            api_key=api_key,
            force_rerun=force_rerun,
            save_activations=self.save_activations
        )


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run SAE evaluations')
    parser.add_argument('--gpu_id', type=str, default="0", help='GPU ID to use')
    parser.add_argument('--sae_key', type=str, default="1", help='sae key to use')
    args = parser.parse_args()

    # Initialize evaluator
    evaluator = SAEEvaluator(
        model_name="pythia-160m",
        gpu_id=args.gpu_id,
        output_dir="eval_results",
        llm_batch_size=100
    )
    
    # Configure SAEs to load
    import json
    sae_config = json.load(open("eval_config/5.8-residual_jumprelu.json"))
    # sae_config = {k: v for k, v in sae_config.items() if args.sae_key in k}

    # Load SAE models (with decoder normalization)
    evaluator.add_saes_from_config(sae_config, normalize=True)
    
    # Run selected evaluations
    eval_types = [
                "core", 
                "sparse_probing",
                "absorption",
                # "autointerp",
                "tpp",
                "scr",
                # "unlearning",
                "ravel"
                ]
    with open("openai_api_key.txt") as f:
        api_key = f.read().strip()
    evaluator.run_evaluations(eval_types=eval_types, api_key=api_key)


if __name__ == "__main__":
    main()