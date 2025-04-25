from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, List, TypeVar
from tqdm import tqdm
from rich import print

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    max_workers: int,
    desc: str = "Processing items",
) -> List[R]:
    """Convenience wrapper for simple threaded mapping with progress bar."""
    results_map = {}  # Store results keyed by future to maintain order if needed (though as_completed doesn't guarantee)
    item_list = list(items)  # Consume iterator for progress bar total

    if not item_list:
        return []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(item_list)}
        # Use tqdm with as_completed for progress updates
        for future in tqdm(as_completed(futures), total=len(item_list), desc=desc):
            original_index = futures[future]
            try:
                result = future.result()
                results_map[original_index] = result
            except Exception as e:
                # Log the error and potentially store a placeholder or skip
                item_info = f"item at index {original_index}"  # Basic info
                # Could try to get more specific item info if item_list[original_index] is simple
                # item_info = str(item_list[original_index])[:50] # Example: first 50 chars
                print(f"[red]Error processing {item_info}: {e}")
                results_map[original_index] = None  # Indicate failure for this item

    # Reconstruct the results list in the original order
    ordered_results: List[R | None] = [
        results_map.get(i) for i in range(len(item_list))
    ]
    # Filter out None values if the caller expects only successful results
    successful_results: List[R] = [res for res in ordered_results if res is not None]

    # Decide whether to return list with Nones or only successful results
    # Returning only successful results seems safer unless None is a valid result of fn
    return successful_results
    # return ordered_results # If None needs to be preserved as failure indicator
