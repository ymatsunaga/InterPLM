""" Embed full protein sequences using ESM model and save to HDF5 file """

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from esm import FastaBatchedDataset, pretrained
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForMaskedLM

from interplm.utils import get_device


def get_model_converter_alphabet(esm_model_name: str, corrupt: bool = False, truncation_seq_length: int = 1022):
    device = get_device()
    _, alphabet = pretrained.load_model_and_alphabet(esm_model_name)
    model = EsmForMaskedLM.from_pretrained(
        f"facebook/{esm_model_name}").to(device)
    model.eval()

    if corrupt:
        model = shuffle_individual_parameters(model)

    batch_converter = alphabet.get_batch_converter(truncation_seq_length)

    return model, batch_converter, alphabet


def shuffle_individual_parameters(model, seed=42):
    """
    Randomly shuffle all parameters within a model while preserving their shapes.

    Args:
        model: PyTorch model to shuffle
        seed: Random seed for reproducibility

    Returns:
        Model with shuffled parameters
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    for param in model.parameters():
        original_shape = param.data.shape
        flat_param = param.data.view(-1)
        shuffled_indices = torch.randperm(flat_param.nelement())
        shuffled_param = flat_param[shuffled_indices]
        param.data = shuffled_param.view(original_shape)

    return model


def embed_list_of_prot_seqs(
    protein_seq_list: List[str],
    esm_model_name: str,
    layer: int,
    toks_per_batch: int = 4096,
    truncation_seq_length: int = None,
    device: torch.device = None,
    corrupt: bool = False
) -> List[np.ndarray]:
    """ Return list of ESM embeddings for a specified layer, computed in batches """
    if device is None:
        device = get_device()

    # Load ESM model
    model, batch_converter, alphabet = get_model_converter_alphabet(
        esm_model_name, corrupt, truncation_seq_length)

    # Create FastaBatchedDataset
    labels = [f"protein_{i}" for i in range(len(protein_seq_list))]
    dataset = FastaBatchedDataset(labels, protein_seq_list)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=batch_converter,
        batch_sampler=batches,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Processing {len(dataset):,} sequences")
    total_tokens = 0
    all_embeddings = [None] * len(protein_seq_list)  # Pre-allocate list

    for labels, strs, toks in tqdm(data_loader, desc="Processing batches"):
        toks = toks.to(device)

        with torch.no_grad():
            results = model(toks, attention_mask=(
                toks != alphabet.padding_idx), output_hidden_states=True)
            embeddings = results.hidden_states[layer]

        # Remove padding and special tokens, and store in the correct position
        for i, (label, seq) in enumerate(zip(labels, strs)):
            seq_len = len(seq)
            # Extract original index from label
            seq_idx = int(label.split('_')[1])
            all_embeddings[seq_idx] = embeddings[i, 1:seq_len+1]

        total_tokens += embeddings.shape[0] * embeddings.shape[1]

    print(f"Processed {total_tokens:,} tokens in total")

    # Verify that all sequences have been processed
    assert all(
        emb is not None for emb in all_embeddings), "Some sequences were not processed"

    return all_embeddings


def embed_single_sequence(
    sequence: str,
    model_name: str,
    layer: int,
    device: torch.device = None
) -> torch.Tensor:
    """
    Embed a single protein sequence using ESM model

    Notably this method doesn't use FastaBatchedDataset or a real dataloader, which
    are more helpful for processing many sequences but doesn't work as well for
    just frequently querying individual sequences quickly and simply as is done
    in the dashboard, so this is easier for that use case (and causes fewer issues
    when multiple users are querying sequences at once).

    Args:
        sequence: Protein sequence string
        model_name: Name of the ESM model to use
        layer: Which layer to extract embeddings from
        device: Torch device to use

    Returns:
        Tensor of embeddings for the sequence
    """
    if device is None:
        device = get_device()

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_name}")
    model = EsmForMaskedLM.from_pretrained(f"facebook/{model_name}")
    model = model.to(device)
    model.eval()

    # Tokenize sequence
    inputs = tokenizer(sequence, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Get embeddings from specified layer
        embeddings = outputs.hidden_states[layer]
        # Remove batch dimension and special tokens
        embeddings = embeddings[0, 1:-1]

    return embeddings
