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

from interplm.esm.embed import get_model_converter_alphabet


def get_activations(
    model: torch.nn.Module,
    batch_tokens: torch.Tensor,
    batch_mask: torch.Tensor,
    layers: List[int]
) -> dict:
    """
    Extract all activation values from multiple layers of the ESM model.

    Takes a batch of tokens, processes them through the model, and returns
    the representations from specified layers. Excludes padding tokens from
    the output. Note, this returns all all tokens flattened together so this
    is only really useful for training, not if you want to keep track of the
    activations for each sequence.

    Args:
        model: ESM model instance
        batch_tokens: Tokenized sequences
        layers: List of layer numbers to extract

    Returns:
        Dictionary mapping layer numbers to their activation tensors
    """
    with torch.no_grad():
        output = model(
            batch_tokens, attention_mask=batch_mask, output_hidden_states=True
        )
        token_representations = {
            layer: output.hidden_states[layer] for layer in layers}

    # Create a mask for non-padding tokens (tokens 0,1,2 are cls/pad/eos respectively)
    mask = batch_tokens > 2
    return {layer: rep[mask] for layer, rep in token_representations.items()}


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
    model, batch_converter, alphabet = get_model_converter_alphabet(
        esm_model_name, corrupt_esm, truncation_seq_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ############ edited ##############
    if weight_file is not None:
        ckpt: Dict[str, Any] = torch.load(weight_file, map_location=device)
        # 典型的な 3 パターンに対応
        if "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        # strict=False で不足 / 余剰キーは無視
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(
            f"Loaded external weights from {weight_file} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )    
    ##################################

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
            "d_model": model.config.hidden_size,
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
