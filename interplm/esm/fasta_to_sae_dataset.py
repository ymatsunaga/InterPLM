"""
Convert a directory of FASTA files to a directories of ESM layer activations organized
by layer and shard with specific metadata used for SAE training.
"""
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from esm import FastaBatchedDataset
from tqdm import tqdm

# from interplm.esm.embed import get_model_converter_alphabet  # Not needed anymore
import esm

def get_activations(
    model: torch.nn.Module,
    batch_tokens: torch.Tensor,
    batch_mask: torch.Tensor,
    layers: List[int],
    dtype: torch.dtype = torch.float32, 
) -> dict:
    """
    Extract activations from multiple ESM layers in a memory-efficient way.

    * 元コードとの主な差分 *
      1. torch.inference_mode() で余計なバッファを生成しない
      2. 取得直後に **CPU へ転送 & 指定 dtype にキャスト** して GPU メモリを開放
      3. 余分なテンソルを確実に捨てるため del と empty_cache() を追加
    """
    with torch.inference_mode():                # ★ 2) no_grad より軽量
        results = model(batch_tokens, repr_layers=layers)

    mask = batch_tokens > 2                     # cls/pad/eos を除外
    activations = {}

    for layer in layers:
        rep = results["representations"][layer]          # GPU 上の fp32
        rep = rep[mask]                                  # パディング除外
        # ★ 3) 取得直後に CPU & 半精度へ退避 → GPU メモリから即座に解放
        activations[layer] = rep.to(dtype=dtype, device="cpu").clone()

    # ★ 4) 不要になった GPU テンソルを確実に解放
    del results
    torch.cuda.empty_cache()

    return activations

def embed_fasta_file_for_all_layers(
    esm_model_name: str,
    fasta_file: Path,
    output_dir: Path,
    layers: List[int],
    shard_num: int,
    corrupt_esm: bool = False,
    toks_per_batch: int = 1024,
    truncation_seq_length: int = 1022,
    weight_file: Path | None = None, 
):
    """
    Process a FASTA file through an ESM model and save layer activations.

    Processes sequences in batches, extracts activations from specified layers,
    shuffles the results, and saves them along with metadata. Uses GPU if available
    and not explicitly disabled.

    Args:
        model: ESM model instance
        alphabet: ESM alphabet for tokenization
        esm_model_name: Name of the ESM model being used
        fasta_file: Path to input FASTA file
        output_dir: Directory to save outputs
        layers: List of layer numbers to extract
        shard_num: Current shard number being processed
        toks_per_batch: Maximum tokens per batch
        truncation_seq_length: Maximum sequence length before truncation

    Outputs:
        - Saves activation tensors as .pt files
        - Saves metadata as JSON files
        - Creates directory structure for outputs
    """
    # Load model using esm.pretrained (to match SFT_hot.ipynb)
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ############ Load custom weights with ESM extraction ##############
    if weight_file is not None:
        from typing import Dict, Any
        ckpt: Dict[str, Any] = torch.load(weight_file, map_location=device)
        # 典型的な 3 パターンに対応
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            # Direct state_dict (like SFT_hot.pt)
            state_dict = ckpt
            
        # Extract ESM part if keys have 'esm.' prefix (from ESM2_TmRegressor)
        if any(k.startswith('esm.') for k in state_dict.keys()):
            print("Extracting ESM part from full model state_dict...")
            esm_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('esm.'):
                    # Remove 'esm.' prefix to match ESM model structure
                    new_key = k[4:]  # Remove 'esm.'
                    esm_state_dict[new_key] = v
            state_dict = esm_state_dict
            
        # strict=False で不足 / 余剰キーは無視
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            f"Loaded external weights from {weight_file} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )    
    ##################################################################

    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=batch_converter,
        batch_sampler=batches,
        num_workers=4,
        pin_memory=True,
    )
    print(f"Read {fasta_file} with {len(dataset):,} sequences")

    total_tokens = 0
    all_activations = {layer: [] for layer in layers}

    for (_, _, toks) in tqdm(data_loader, desc="Processing batches"):
        activations = get_activations(model,
                                      toks.to(device),
                                      (toks != alphabet.padding_idx).to(device),
                                      layers=layers)
        for layer in layers:
            all_activations[layer].append(activations[layer])

        # Count total tokens processed
        total_tokens += activations[layers[0]].shape[0]

        torch.cuda.empty_cache()

    # Save activations and metadata for each layer in the proper directory structure
    for layer in layers:
        layer_output_dir = output_dir / f"layer_{layer}" / f"shard_{shard_num}"
        layer_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = layer_output_dir / "activations.pt"
        metadata_file = layer_output_dir / "metadata.json"

        # Concatenate all activations for this layer
        layer_activations = torch.cat(all_activations[layer])

        # Shuffle the activations
        shuffled_indices = torch.randperm(total_tokens)
        layer_activations = layer_activations[shuffled_indices]

        # Save the tensor
        torch.save(layer_activations, output_file)
        print(f"Saved activations for layer {layer}, shard {shard_num} to {output_file}")

        # Save metadata
        metadata = {
            "model": esm_model_name,
            "total_tokens": total_tokens,
            "d_model": model.embed_dim,
            "dtype": str(layer_activations.dtype),
            "layer": layer,
            "shard": shard_num,
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)


def process_shard_range(
    fasta_dir: Path,
    output_dir: Path = Path("../../data/embeddings"),
    esm_model_name: str = "esm2_t6_8M_UR50D",
    layers: List[int] = [1, 2, 3, 4, 5, 6],
    start_shard: int | None = None,
    end_shard: int | None = None,
    corrupt_esm: bool = False,
    weight_file: Path | None = None,
):
    """
    Process a range of FASTA shards through an ESM model.

    Processes each shard in the specified range, extracting and saving activations
    from specified layers. Can optionally use a corrupted model with shuffled
    parameters. Skips shards that have already been processed.

    Args:
        start_shard: First shard number to process
        end_shard: Last shard number to process (inclusive)
        esm_model_name: Name of the ESM model to use
        layers: List of layer numbers to extract
        fasta_dir: Directory containing FASTA shard files
        output_dir: Directory to save outputs
        corrupt_esm: Whether to shuffle model parameters

    Outputs:
        Creates directory structure with:
        - Activation tensors for each layer
        - Metadata files for each processed shard
    """

    # identify the number of shards in the fasta_dir
    fasta_files = list(fasta_dir.glob("*.fasta"))
    if not fasta_files:
        raise ValueError(f"No FASTA files found in {fasta_dir}")

    if start_shard is None:
        start_shard = 0
    if end_shard is None:
        end_shard = len(fasta_files) - 1

    for i in range(start_shard, end_shard + 1):
        embed_fasta_file_for_all_layers(
            esm_model_name=esm_model_name,
            corrupt_esm=corrupt_esm,
            fasta_file=fasta_dir / f"shard_{i}.fasta",
            output_dir=output_dir,
            layers=layers,
            shard_num=i,
            weight_file=weight_file, 
        )


if __name__ == "__main__":
    from tap import tapify
    tapify(process_shard_range)
