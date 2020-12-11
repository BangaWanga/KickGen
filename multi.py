import multiprocessing as mp
import logging
from datetime import datetime
import psutil
from threading import Lock


class data_handler:
    def __init__(self):
        pass

    def multi_threading(self, iterable, function):
        generator = self._make_gen(iterable)
        with mp.Manager() as manager:

            return_list = manager.list()
            mutex = mp.Lock()

            logging.debug("Start spawning Multiprocesses")
            seq = generator()
            procs = list()
            while True:
                try:

                    n_cpus = psutil.cpu_count()
                    for cpu in range(n_cpus):
                        next_item = next(seq)
                        affinity = [cpu]
                        p = mp.Process(target=self.run_child, args=(mutex, return_list, affinity, function, next_item))
                        p.start()
                        procs.append(p)
                except StopIteration:
                    for p in procs:
                        p.join()
                    logging.debug("Finished Multiprocessing")
                    l = list(return_list)
                    del return_list
                    return l
                except Exception as e:
                    del mutex


    @staticmethod
    def _make_gen(iterable):
        def gen():
            while iterable:
                yield iterable.pop()
        return gen

    @staticmethod
    def run_child(mutex, shared_list, affinity, f, next_item):
        done = False
        while not done:
            try:
                proc = psutil.Process()
                proc.cpu_affinity(affinity)
                _aff = proc.cpu_affinity()
                item = f(next_item)
                with mutex:
                    shared_list.append(item)
                done = True
            except mp.managers.RemoteError as e:
                logging.error(e)
            except OSError as e:
                raise


if __name__ == '__main__':
    mp.freeze_support()