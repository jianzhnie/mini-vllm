"""Multiprocessing Event synchronization demo.

Shows how to use multiprocessing.Event for graceful shutdown of worker
processes, which is the same pattern used in mini-vLLM's tensor parallel
workers.

Usage:
    python examples/mp_event_demo.py
"""

import multiprocessing
import time


def worker(stop_event):
    """Worker process that runs until the event is set."""
    print('Worker started...')
    while not stop_event.is_set():
        print('Working... +1')
        # wait(timeout) is preferred over time.sleep() because it
        # responds immediately when the event is set
        stop_event.wait(timeout=1)

    print('Worker received stop signal, shutting down.')


if __name__ == '__main__':
    stop_event = multiprocessing.Event()

    p = multiprocessing.Process(target=worker, args=(stop_event,))
    p.start()

    # Let the worker run for 3 seconds
    time.sleep(3)

    print('Main process: sending stop signal')
    stop_event.set()

    p.join()
    print('Main process: all workers stopped')
