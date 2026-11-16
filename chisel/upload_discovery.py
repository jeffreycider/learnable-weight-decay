"""Bounded receipt discovery that never blocks upload queue replenishment."""
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty, Full
from threading import Event
import time
from chisel.receipt_discovery import ready_receipts


class Inventory:
    def __init__(self, root, pattern, interval, capacity=2048):
        self.root, self.pattern, self.interval = root, pattern, interval
        self.queue, self.stopping = Queue(capacity), Event()
        self.pool = ThreadPoolExecutor(1)
        self.future, self.next_scan = None, 0.
        self.receipts, self.seen = [], set()

    def scan(self):
        receipts = ready_receipts(self.root) if self.pattern == '**/*.ready.json' else self.root.glob(self.pattern)
        for receipt in receipts:
            while not self.stopping.is_set():
                try:
                    self.queue.put(receipt, timeout=.1)
                    break
                except Full:
                    continue
            if self.stopping.is_set():
                break

    def poll(self):
        if self.future is not None and self.future.done():
            self.future.result()
            self.future = None
            self.next_scan = time.monotonic() + self.interval
        if self.future is None and time.monotonic() >= self.next_scan:
            self.future = self.pool.submit(self.scan)
        for _ in range(self.queue.maxsize):
            try:
                receipt = self.queue.get_nowait()
            except Empty:
                break
            if receipt not in self.seen:
                self.seen.add(receipt)
                self.receipts.append(receipt)
        return self.receipts

    def close(self):
        self.stopping.set()
        self.pool.shutdown(wait=True)
