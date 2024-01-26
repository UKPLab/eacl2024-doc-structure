from __future__ import annotations

from abc import ABC
from ctypes import c_int
from multiprocessing import Manager, Pool
import os
from pathlib import Path
from shutil import rmtree
from typing import List, Optional, Union

from intertext_graph.itgraph import IntertextDocument
from intertext_graph.parsers.utils import chunksize, num_processes
from tqdm import tqdm


class Pipeline(ABC):

    def __init__(self, *pipeline: str, out: Path, max_instances: int = None) -> None:
        self._pipeline = pipeline
        self.out = out
        if max_instances:
            manager = Manager()
            self._max_instances = manager.Value(c_int, max_instances)
            self._max_instances_lock = manager.Lock()
        self._desc = 'generic'

    def _save(self, doc: IntertextDocument) -> None:
        raise NotImplemented

    def _instance_done(self) -> None:
        if hasattr(self, '_max_instances'):
            with self._max_instances_lock:
                self._max_instances.value -= 1

    @property
    def _max_instances_done(self) -> bool:
        return hasattr(self, '_max_instances') and self._max_instances.value <= 0

    def _batch_func(self, data: Union[Path, IntertextDocument]) -> None:
        if self._max_instances_done:
            return
        try:
            for func in self._pipeline:
                tmp = getattr(self, f'_{func}')(data)
                # Use a boolean return type to decide whether to continue or skip this doc
                # Can be used for filter operations
                if isinstance(tmp, bool) or tmp is None:
                    if tmp:
                        continue
                    else:
                        return
                assert isinstance(tmp, IntertextDocument)
                data = tmp
            self._save(data)
            self._instance_done()
        except AssertionError as err:
            raise err
        except Exception as err:
            print(err)

    def __call__(self, lst: List[Path | str | IntertextDocument], reset: bool = True) -> None:
        if reset:
            rmtree(self.out, ignore_errors=True)
            self.out.mkdir(parents=True)
        total = len(lst)
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        with Pool(processes=num_processes()) as pool:
            for _ in tqdm(
                    pool.imap_unordered(self._batch_func, lst, chunksize(total)),
                    total=total,
                    desc=f'running {self._desc} pipeline'):
                if self._max_instances_done:
                    pool.terminate()
                    break
            pool.close()
            pool.join()
        os.unsetenv('TOKENIZERS_PARALLELISM')

    @classmethod
    def run_default_config(cls, *args, **kwargs) -> Optional[Pipeline]:
        raise NotImplemented

