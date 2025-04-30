import json
import multiprocessing
import queue  # For QueueEmpty exception
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,  # Add this import back
    as_completed,
)
from pathlib import Path
import json
import logging  # Import logging
import multiprocessing
import queue  # For QueueEmpty exception
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,  # Add this import back
    as_completed,
)
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, TypeVar

# from rich.console import Console # Remove Console import
from rich.progress import Progress
from tqdm import tqdm  # Add this import back for parallel_map

T = TypeVar("T")
R = TypeVar("R")

# Get logger for this module
logger = logging.getLogger(__name__)


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
                logger.error(f"Error processing {item_info}: {e}")
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


# --- New function for incremental saving ---

_SENTINEL = None  # Signal for the writer process to stop


def _writer_process(
    output_queue: multiprocessing.Queue, output_file: Path, total_items: int
):
    """Target function for the writer process."""
    processed_count = 0
    try:
        with (
            output_file.open(
                "w", encoding="utf-8"
            ) as f,  # Start fresh or append? Let's start fresh.
            Progress() as progress,
        ):
            task = progress.add_task(
                f"[cyan]Writing results to {output_file.name}...", total=total_items
            )
            while True:
                try:
                    # Wait for a short time to avoid busy-waiting if queue is empty
                    result = output_queue.get(timeout=0.1)
                    if result is _SENTINEL:
                        # Ensure progress bar completes if sentinel received early
                        progress.update(task, completed=total_items)
                        break  # Exit loop when sentinel is received

                    # Write the received result (should be a dictionary) as JSON line
                    f.write(json.dumps(result) + "\n")
                    processed_count += 1
                    progress.update(task, advance=1)

                except queue.Empty:
                    # Queue is empty, continue waiting
                    continue
                except (EOFError, BrokenPipeError):
                    # Parent process might have exited unexpectedly
                    logger.error(
                        "Writer process: Communication pipe broken. Exiting."
                    )
                    break
                except Exception as e:
                    # Log errors during writing but try to continue
                    logger.error(f"Writer process error writing result: {e}")
                    # Optionally, write the error to a separate log?
                    # For now, just log and advance progress bar to avoid stall
                    processed_count += 1
                    progress.update(task, advance=1)

    except Exception as e:
        logger.exception(
            f"Writer process failed to open/write file {output_file}: {e}"
        )
    finally:
        logger.info(
            f"Writer process finished. Items written: {processed_count}/{total_items}"
        )


def _worker_wrapper(
    fn: Callable[[Any], Dict[str, Any]], item: Any, output_queue: multiprocessing.Queue
):
    """Wrapper to run the target function and put the result on the queue."""
    try:
        result = fn(item)
        output_queue.put(result)
    except Exception as e:
        # Log error and potentially put an error object on the queue if needed
        logger.error(f"Worker error processing item {item}: {e}")
        # Decide if you want to signal failure via the queue.
        # For now, we assume fn handles its own errors and returns a dict with an error field.
        # If fn raises, the result won't be put on the queue.


def parallel_map_and_save(
    fn: Callable[[Any], Dict[str, Any]],
    items: Iterable[Any],
    output_file: Path,
    max_workers: int,
    desc: str = "Processing items",
):
    """
    Processes items in parallel using ProcessPoolExecutor and saves results
    incrementally to a JSON Lines file via a dedicated writer process.

    Args:
        fn: The function to apply to each item. Must return a dictionary.
        items: An iterable of items to process.
        output_file: The Path object for the output JSON Lines file.
        max_workers: The maximum number of worker processes.
        desc: Description for the overall progress bar (not used for writer).
    """
    items_list = list(
        items
    )  # Need length and ability to iterate multiple times if needed
    total_items = len(items_list)
    if total_items == 0:
        logger.info("No items to process.")
        # Ensure empty file is created? Or do nothing? Let's do nothing.
        # output_file.touch() # Optionally create empty file
        return

    # Use a Manager queue for ProcessPoolExecutor compatibility
    manager = multiprocessing.Manager()
    output_queue = manager.Queue()

    # Start the writer process
    writer = multiprocessing.Process(
        target=_writer_process, args=(output_queue, output_file, total_items)
    )
    writer.start()

    # Use ProcessPoolExecutor to run worker wrappers
    processed_count = 0
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks using the wrapper
            futures = [
                executor.submit(_worker_wrapper, fn, item, output_queue)
                for item in items_list
            ]

            # Optional: Track overall progress if needed (distinct from writer progress)
            # with Progress() as progress:
            #     task = progress.add_task(f"[cyan]{desc}...", total=total_items)
            #     for future in as_completed(futures):
            #         # We don't need the result here, it went to the queue
            #         processed_count += 1
            #         progress.update(task, advance=1)
            #         # Handle potential exceptions raised by the worker *wrapper* itself (rare)
            #         try:
            #             future.result() # Check for exceptions from the submit/wrapper
            #         except Exception as e:
            #             print(f"[red]Error in worker future: {e}")

            # Simpler wait without overall progress bar:
            for future in as_completed(futures):
                processed_count += 1
                try:
                    future.result()  # Check for exceptions from the submit/wrapper
                except Exception as e:
                    logger.error(f"Error retrieving worker future result: {e}")

    except Exception as e:
        logger.exception(f"Error during parallel execution: {e}")
        # Consider how to signal the writer to stop cleanly in case of executor error
        # For now, it might hang or exit due to broken pipe
    finally:
        # Signal the writer process to finish
        logger.info("Signaling writer process to stop...")
        output_queue.put(_SENTINEL)

        # Wait for the writer process to terminate
        logger.info("Waiting for writer process to finish...")
        writer.join(timeout=10)  # Add a timeout
        if writer.is_alive():
            logger.warning("Writer process did not exit cleanly. Terminating.")
            writer.terminate()
            writer.join()

        manager.shutdown()  # Shutdown the manager
        logger.info(
            f"Parallel processing finished. Workers attempted: {processed_count}"
        )
