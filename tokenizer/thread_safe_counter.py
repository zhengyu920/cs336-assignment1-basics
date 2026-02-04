from multiprocessing.managers import SyncManager

class ThreadSafeCounter:
    def __init__(self, manager: SyncManager):
        self._data = manager.dict()
        self._lock = manager.Lock()

    def increment(self, key, amount=1):
        with self._lock:
            self._data[key] = self._data.get(key, 0) + amount
    
    def increment_from_dict(self, counter):
        with self._lock:
            for key, count in counter.items():
                self._data[key] = self._data.get(key, 0) + count

    def to_dict(self):
        """Returns a standard, local Python dictionary snapshot."""
        with self._lock:
            # Converting the DictProxy to a standard dict 
            # effectively "copies" the data into the local process memory.
            return dict(self._data)

    def __getitem__(self, key):
        """Allows syntax like: counter['hits']"""
        with self._lock:
            return self._data.get(key, 0)