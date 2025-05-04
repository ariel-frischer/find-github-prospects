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

import litellm  # Add import for type hinting

# from rich.console import Console # Remove Console import
from rich.progress import Progress
from tqdm import tqdm

T = TypeVar("T")
R = TypeVar("R")

# Type hint for the shared dictionary
CostDataType = Dict[str, Any]

# Get logger for this module
logger = logging.getLogger(__name__)

# --- Manual Cost Calculation ---
# Costs are per 1,000,000 tokens (divide by 1e6 for per-token cost)
# Source: User provided Gemini 2.5 Flash/Pro costs
# Ensure model names here EXACTLY match those returned by LiteLLM/OpenRouter
# (e.g., in response.model) and used in config.py/url_processor.py
MODEL_COST_MAP = {
    # Summarizer Model (Gemini Flash 2.5 Preview via OpenRouter)
    "openrouter/google/gemini-2.5-flash-preview": {  # Updated model name
        "input_cost_per_token": 0.15 / 1_000_000,  # Cost per user request
        "output_cost_per_token": 0.60 / 1_000_000,  # Cost per user request
    },
    # Summarizer Model (Gemini Flash 2.5 Preview via Google directly)
    "google/gemini-2.5-flash-preview": {  # Added direct Google model
        "input_cost_per_token": 0.15 / 1_000_000,
        "output_cost_per_token": 0.60 / 1_000_000,
    },
    # Enricher Model (Gemini Pro 2.5 Preview via OpenRouter)
    "openrouter/google/gemini-2.5-pro-preview-03-25": {
        "input_cost_per_token": 1.25 / 1_000_000,
        "output_cost_per_token": 10.00 / 1_000_000,
    },
    # Enricher Model (Gemini Pro 2.5 Preview via Google directly)
    "google/gemini-2.5-pro-preview-03-25": {  # Added direct Google model
        "input_cost_per_token": 1.25 / 1_000_000,
        "output_cost_per_token": 10.00 / 1_000_000,
    },
    # Add other models used here if necessary
}


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


# --- Cost Tracking Helper ---


def _update_shared_cost(
    response: litellm.utils.ModelResponse,  # Use ModelResponse for type hint
    shared_cost_data: CostDataType,
    lock: multiprocessing.Lock,
    call_description: str,
):
    """
    Safely updates shared cost/token data and logs individual call details using manual calculation.
    """
    try:
        # Get model name and usage from response
        model_name = getattr(response, "model", None)
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0)
        output_tokens = getattr(usage, "completion_tokens", 0)

        if not model_name or usage is None:
            logger.warning(
                f"    Could not get model name or usage from response for {call_description}. Cannot calculate cost."
            )
            return

        # Look up costs in the map
        cost_info = MODEL_COST_MAP.get(model_name)

        if cost_info:
            # Calculate cost manually
            input_cost = input_tokens * cost_info["input_cost_per_token"]
            output_cost = output_tokens * cost_info["output_cost_per_token"]
            total_call_cost = input_cost + output_cost

            logger.info(
                f"    LLM Call Cost ({call_description} - {model_name}): ${total_call_cost:.6f} "
                f"(In: {input_tokens}, Out: {output_tokens})"
            )

            # Update shared data
            with lock:
                shared_cost_data["total_cost"] = (
                    shared_cost_data.get("total_cost", 0.0) + total_call_cost
                )
                shared_cost_data["total_input_tokens"] = (
                    shared_cost_data.get("total_input_tokens", 0) + input_tokens
                )
                shared_cost_data["total_output_tokens"] = (
                    shared_cost_data.get("total_output_tokens", 0) + output_tokens
                )
        else:
            logger.warning(
                f"    Cost info not found in MODEL_COST_MAP for model: {model_name}. Cannot calculate cost for {call_description}."
            )
            # Still update token counts even if cost is unknown
            with lock:
                shared_cost_data["total_input_tokens"] = (
                    shared_cost_data.get("total_input_tokens", 0) + input_tokens
                )
                shared_cost_data["total_output_tokens"] = (
                    shared_cost_data.get("total_output_tokens", 0) + output_tokens
                )

    except AttributeError as ae:
        logger.warning(
            f"    Attribute error accessing response data for {call_description}: {ae}"
        )
    except Exception as e:
        logger.warning(
            f"    Error processing cost/token data for {call_description}: {e}"
        )


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
                    logger.error("Writer process: Communication pipe broken. Exiting.")
                    break
                except Exception as e:
                    # Log errors during writing but try to continue
                    logger.error(f"Writer process error writing result: {e}")
                    # Optionally, write the error to a separate log?
                    # For now, just log and advance progress bar to avoid stall
                    processed_count += 1
                    progress.update(task, advance=1)

    except Exception as e:
        logger.exception(f"Writer process failed to open/write file {output_file}: {e}")
    finally:
        logger.info(
            f"Writer process finished. Items written: {processed_count}/{total_items}"
        )


def _worker_wrapper(
    fn: Callable[
        [Any, CostDataType, multiprocessing.Lock], Dict[str, Any]
    ],  # Add shared types to signature
    item: Any,
    output_queue: multiprocessing.Queue,
    shared_cost_data: CostDataType,  # Add shared dict
    lock: multiprocessing.Lock,  # Add lock
):
    """Wrapper to run the target function and put the result on the queue."""
    try:
        # Pass shared data and lock to the target function
        result = fn(item, shared_cost_data, lock)
        output_queue.put(result)
    except Exception as e:
        # Log error and potentially put an error object on the queue if needed
        logger.error(f"Worker error processing item {item}: {e}")
        # Decide if you want to signal failure via the queue.
        # For now, we assume fn handles its own errors and returns a dict with an error field.
        # If fn raises, the result won't be put on the queue.


def parallel_map_and_save(
    fn: Callable[
        [Any, CostDataType, multiprocessing.Lock], Dict[str, Any]
    ],  # Update fn signature hint
    items: Iterable[Any],
    output_file: Path,
    max_workers: int,
    shared_cost_data: CostDataType,  # Add shared dict parameter
    lock: multiprocessing.Lock,  # Add lock parameter
    desc: str = "Processing items",  # This desc is unused now
):
    """
    Processes items in parallel using ProcessPoolExecutor, saves results
    incrementally, and aggregates cost/token data using shared objects.

    Args:
        fn: The function to apply to each item. Must accept item, shared_cost_data, lock
            and return a dictionary.
        items: An iterable of items to process.
        output_file: The Path object for the output JSON Lines file.
        max_workers: The maximum number of worker processes.
        shared_cost_data: The shared dictionary for accumulating costs/tokens.
        lock: The shared lock for accessing shared_cost_data.
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
            # Submit tasks using the wrapper, passing shared objects
            futures = [
                executor.submit(
                    _worker_wrapper, fn, item, output_queue, shared_cost_data, lock
                )
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
