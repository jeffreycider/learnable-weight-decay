"""Bounded, immutable teacher logits keyed by checkpoint and exact replayed examples."""
from collections import OrderedDict


class TeacherCache:
    def __init__(self, maximum_bytes):
        self.maximum_bytes = maximum_bytes
        self.values, self.bytes = OrderedDict(), 0
        self.hits = self.misses = 0

    def get(self, key):
        if key is None or key not in self.values:
            self.misses += 1
            return None
        value = self.values.pop(key)
        self.values[key] = value
        self.hits += 1
        return value

    def put(self, key, value):
        size = value.numel() * value.element_size()
        if key is None or size > self.maximum_bytes:
            return
        if key in self.values:
            previous = self.values.pop(key)
            self.bytes -= previous.numel() * previous.element_size()
        while self.values and self.bytes + size > self.maximum_bytes:
            previous = self.values.popitem(last=False)[1]
            self.bytes -= previous.numel() * previous.element_size()
        self.values[key] = value.detach().clone()
        self.bytes += size
