import random
import re
import time

from repobird_leadgen.utils import parallel_map

# ANSI escape code pattern
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*[mK]")


def strip_ansi_codes(text):
    """Removes ANSI escape codes from a string."""
    # First, handle the raw escape characters captured by pytest
    text = text.replace("\x1b", "")
    # Then, use regex to remove the ANSI sequences
    return re.sub(r"\[[0-9;]*[mK]", "", text)


# Helper function for successful execution
def square(x):
    # Simulate some work
    time.sleep(random.uniform(0.01, 0.05))
    return x * x


# Helper function that raises an exception
def raise_exception(x):
    if x == 3:
        raise ValueError("Test exception for value 3")
    # Simulate work for non-failing items
    time.sleep(random.uniform(0.01, 0.05))
    return x


# Helper function that simulates long processing for one item
def slow_square(x):
    if x == 2:
        time.sleep(0.2)  # Simulate long work for item '2'
    else:
        time.sleep(random.uniform(0.01, 0.05))
    return x * x


def test_parallel_map_success():
    """Test parallel_map with a function that completes successfully."""
    inputs = [1, 2, 3, 4, 5]
    expected_results = [1, 4, 9, 16, 25]
    results = parallel_map(square, inputs, max_workers=4)
    assert results == expected_results


def test_parallel_map_exception_handling(caplog):
    """Test parallel_map with a function that raises an exception for some inputs."""
    inputs = [1, 2, 3, 4, 5]
    expected_results = [1, 2, 4, 5]
    results = parallel_map(raise_exception, inputs, max_workers=2)
    assert results == expected_results

    # Check that the error was logged (captured by caplog)
    assert "Error processing item at index 2" in caplog.text
    assert "Test exception for value 3" in caplog.text


def test_parallel_map_empty_input():
    """Test parallel_map with an empty input list."""
    inputs = []
    results = parallel_map(square, inputs, max_workers=2)
    assert results == []


def test_parallel_map_order_preserved_despite_processing_time():
    """Test that result order corresponds to input order, not completion order."""
    inputs = [1, 2, 3, 4]
    expected_results = [1, 4, 9, 16]
    results = parallel_map(slow_square, inputs, max_workers=4)
    assert results == expected_results


def test_parallel_map_single_worker():
    """Test parallel_map behaves correctly when max_workers=1 (sequentially)."""
    inputs = [1, 2, 3, 4, 5]
    expected_results = [1, 4, 9, 16, 25]
    results = parallel_map(square, inputs, max_workers=1)
    assert results == expected_results
