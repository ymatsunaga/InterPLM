import re
from logging import getLogger
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pandas as pd

logger = getLogger(__name__)


def process_index_range(index: str, seq_len: int) -> Tuple[Optional[int], Optional[int]]:
    """
    Process a single index or range from UniProt annotation.

    Args:
        index: String containing either a single index or range (e.g., "5" or "3..7")
        seq_len: Length of the protein sequence

    Returns:
        Tuple of (start_index, end_index). For single indices, end_index will be None.
    """
    if ":" in index or "?" in index:
        return None, None

    try:
        if ".." in index:
            start, end = index.split("..")
            start = int(start.strip("<")) - 1  # Convert to 0-based indexing
            end = min(int(end.strip(">")), seq_len)
            return start, end
        else:
            idx = int(index) - 1
            return (idx, None) if 0 <= idx < seq_len else (None, None)
    except Exception as e:
        logger.warning(f"Error processing index {index}: {e}")
        return None, None


def process_feature_entries(column_data: str,
                            column_name: str,
                            seq_len: int,
                            process_fn: Callable) -> Any:
    """
    Generic function to process feature entries from UniProt annotations.

    Args:
        column_data: Raw feature data from UniProt
        column_name: Name of the feature column
        seq_len: Length of the protein sequence
        process_fn: Function to process each entry

    Returns:
        Result from the process_fn
    """
    if pd.isna(column_data):
        return process_fn(None, None, True)

    entries = column_data.split(f"{column_name} ")[1:]

    for entry in entries:
        index = entry.split(";")[0]
        start, end = process_index_range(index, seq_len)
        if not process_fn(start, end, False):
            break

    return process_fn(None, None, True)


def process_binary_feature(column_data: str,
                           column_name: str,
                           seq_len: int,
                           current_index: int) -> Tuple[List[bool], int]:
    """
    Process binary features from UniProt annotations.
    """
    indices = [False] * seq_len

    def process_entry(start: Optional[int],
                      end: Optional[int],
                      is_final: bool) -> bool:
        nonlocal current_index

        if is_final:
            return indices, current_index

        if start is not None:
            if end is not None:
                for i in range(start, end):
                    indices[i] = current_index
            else:
                indices[start] = current_index
            current_index += 1
        return True

    return process_feature_entries(column_data, column_name, seq_len, process_entry)


def process_interaction_feature(column_data: str,
                                column_name: str,
                                seq_len: int) -> Tuple[List[bool], List[Optional[int]]]:
    """
    Process interaction features (like disulfide bonds) from UniProt annotations.
    """
    indices = [False] * seq_len
    pairs = [None] * seq_len
    n_pairs = 0

    def process_entry(start: Optional[int],
                      end: Optional[int],
                      is_final: bool) -> bool:
        nonlocal n_pairs

        if is_final:
            return indices, pairs

        if start is not None:
            indices[start] = True
            if end is not None:
                indices[end - 1] = True
                pairs[start] = n_pairs
                pairs[end - 1] = n_pairs
                n_pairs += 1
        return True

    return process_feature_entries(column_data, column_name, seq_len, process_entry)


def analyze_categorical_features(df: pd.DataFrame,
                                 category: str,
                                 category_name: str,
                                 separator_name: str = "note") -> Tuple[int, pd.Series, List[int], pd.Series]:
    """
    Analyze categorical features to find common categories and their statistics.
    """
    non_na = df[category].dropna()
    n_per_prot = non_na.apply(lambda x: x.count(category_name)).value_counts()

    all_notes = []
    all_lengths = []

    def process_entry(entry: str) -> None:
        note_start = entry.find(separator_name)
        note = entry[note_start + len(separator_name) + 1:]
        note_end = note.find(';')
        note = note[:note_end].strip('"')

        if "/evidence" not in note:
            all_notes.append(note)

        try:
            location = entry[1:]
            end_of_loc = location.find(";")
            location = location[:end_of_loc] if end_of_loc != -1 else location

            start, end = process_index_range(location, float('inf'))
            if start is not None:
                if end is not None:
                    all_lengths.append(end - start)
                else:
                    all_lengths.append(1)

        except Exception as e:
            logger.warning(f"Error analyzing feature length: {e}")

    for row in non_na:
        entries = row.split(category_name)[1:]
        for entry in entries:
            process_entry(entry)

    return (
        len(non_na),
        n_per_prot.sort_values(ascending=False),
        all_lengths,
        pd.Series(all_notes).value_counts()
    )


def process_categorical_feature(
    column_data: str,
    column_name: str,
    category_options: Set[str],
    seq_len: int,
    current_index: Dict[str, int]
) -> tuple[list, dict]:
    """
    Process categorical features from UniProt annotations.

    Args:
        column_data: Raw feature data from UniProt
        column_name: Name of the feature column
        category_options: Set of valid category names
        seq_len: Length of the protein sequence
        current_index: Dictionary tracking current index for each category

    Returns:
        Tuple of (list of category indices vectors, updated current_index)
    """
    category_indices = {
        category_name: [False] * seq_len for category_name in category_options
    }

    def extract_category(entry: str) -> Optional[str]:
        """Extract category from entry's note field."""
        match = re.search(r'/note="([^"]+)"', entry)
        if match:
            category = match.group(1).split(";")[0]
            return category if category in category_options else None
        return None

    def process_entry(start: Optional[int],
                      end: Optional[int],
                      is_final: bool) -> bool:
        """Process a single entry with its position and category."""
        if is_final:
            return [category_indices[category] for category in category_options], current_index

        if start is not None and entry_category is not None:
            if end is not None:
                for pos in range(start, end):
                    if 0 <= pos < seq_len:
                        category_indices[entry_category][pos] = current_index[entry_category]
                        category_indices["any"][pos] = current_index["any"]
            else:
                if 0 <= start < seq_len:
                    category_indices[entry_category][start] = current_index[entry_category]
                    category_indices["any"][start] = current_index["any"]

            current_index[entry_category] += 1
            current_index["any"] += 1
        return True

    if pd.isna(column_data):
        return [category_indices[category] for category in category_options], current_index

    entries = column_data.split(f"{column_name} ")[1:]

    for entry in entries:
        entry_category = extract_category(entry)
        if entry_category:
            result = process_feature_entries(
                entry, column_name, seq_len, process_entry)
            if result:
                return result

    return [category_indices[category] for category in category_options], current_index
