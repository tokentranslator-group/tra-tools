# $ python3 -c "from tra_tools.progress.progress_mpi import run;run()"
import multiprocessing as mp
import threading
import time

# from progress.prgress_cmd import ProgressCmd
import tra_tools.progress as progress_cmd
ProgressCmd = progress_cmd.ProgressCmd


class ProgressMasterBase():
    '''Used to orchestrate workers progresses.
    Contain the main task (self.run) and
    actual progress to be shown.
    Implementation of the process itself should be
    given in descendants
    '''
    def __init__(self, threads_count, steps):

        mp_context = "spawn"
        self.ctx = mp.get_context(mp_context)

        # actual progress:
        self.progress = ProgressCmd(steps)

        # cache to store threads progress results:
        self.cache_states = [0]*threads_count

        # to detect when job is done:
        self.cache_exiting = [False]*threads_count

    def __enter__(self, *args, **kwargs):
        self.process.start()
        return(self)

    def __exit__(self, *args):
        # self.process.terminate/join?

        # The `join` here is necessery since We want to
        # wait until the `run` method will terminate i.e. until
        # its `while True` loop will be over.
        # Omiting this line and main program will
        # not wait until this happend and just exit.
        # With this exit it automatically kill any
        # proccesses having been started by it
        # so unless `time.sleep()` is not used no
        # results will be produced
        self.process.join(1)
        pass

    def __call__(self):
        # self.process.start()
        return self

    def run(self):

        # TODO: try/finally to catch errors
        while True:
            if self.queue.empty():
                continue
            msg = self.queue.get(timeout=1)
            # try:
            # msg = queue1.get(timeout=1)
            # msg = self.queue.get(timeout=1)
            # except self.queue.Empty:
            #     continue

            thread_idx, progress_idx = msg
            if (progress_idx is None
                or progress_idx >= self.progress.steps_total):
                self.cache_exiting[thread_idx] = True
            else:
                # main:
                self.cache_states[thread_idx] = progress_idx
                self.progress.succ(min(self.cache_states))
                
            # exiting:
            if all(self.cache_exiting):
                self.progress.print_end()
                break
            

class ProgressMasterProcess(ProgressMasterBase):
    '''Implementation with use of mp.Process
    # due to some multiprocessing bug it only
    # work with Manager.Queue:
    # REF: https://stackoverflow.com/questions/73384531/queue-objects-should-only-be-shared-between-processes-through-inheritance-even-w#
    '''
    def __init__(self, *args, **kwargs):
        
        ProgressMasterBase.__init__(self, *args, **kwargs)

        # queue to send progresses steps from each
        # thread to master:
        m = self.ctx.Manager()
        self.queue = m.Queue()

        self.process = mp.Process(target=self.run)
        # self.process = self.ctx.Process(target=run, args=(self.queue,))

    def __exit__(self, *args):
        ProgressMasterBase.__exit__(self, *args)
        self.process.terminate()


class ProgressMasterThread(ProgressMasterBase):

    '''Implementation with use of threading.Thread'''

    def __init__(self, *args, **kwargs):
        
        ProgressMasterBase.__init__(self, *args, **kwargs)
        
        # queue to send progresses steps from each
        # thread to master:
        self.queue = self.ctx.Queue()

        # master process:
        self.process = threading.Thread(target=self.run)
        # self.process.daemon = True

    
class ProgressWorker(ProgressCmd):
    '''The instances of this type is progresses which
    will be called at workers as such. But instead of
    showing they send the data to the master through queue.'''

    def __init__(self, queue, thread_idx):
        self.queue = queue
        self.thread_idx = thread_idx

    def succ(self, idx):
        # print("putting to queue")
        self.queue.put_nowait((self.thread_idx, idx))

    def print_end(self):
        '''Notify the queue/master about finishing progress'''
        self.queue.put_nowait((self.thread_idx, None))

    # for compatibility with some libs:
    def __call__(self, steps):
        self.set_steps(steps)
        return self


# ================= TESTS ====================:
def wrapper(queue, thread_idx, steps):

    '''Example task for workers to solve.'''

    progress = ProgressWorker(queue, thread_idx)
    for idx in range(steps):
        #  print("succ")
        progress.succ(idx)
    progress.print_end()
    return("done thread: %d" % thread_idx)


def test_as_thread():
    cpu_count = mp.cpu_count()
    steps = 10000

    # pool = mp.Pool(processes=cpu_count)
    with ProgressMasterThread(cpu_count, steps)() as p:
        q = p.queue
        pool = p.ctx.Pool(processes=cpu_count)
        print("starting")
        workers = [pool.Process(name=str(i), target=wrapper, args=(q, i, steps)) for i in range(cpu_count)]
        
        # this seems do nothing until
        # master call join
        for w in workers:
            w.start()
        # this will work even without join
        # time.sleep(3)
    for w in workers:
        w.terminate()
    # print(res)
        

def test_as_process():
    cpu_count = mp.cpu_count()
    steps = 1000

    # impossible with using this like
    # see ProgressMasterProcess desctiption:
    # q = mp.Queue()
    
    # pool = mp.Pool(processes=cpu_count)
    with ProgressMasterProcess(cpu_count, steps)() as p:
        q = p.queue
        pool = p.ctx.Pool(processes=cpu_count)
        print("starting")
        res = [pool.apply_async(wrapper, (q, i, steps)) for i in range(cpu_count)]
        res = [r.get() for r in res]
    print(res)


def run():
    print("test_as_process:")
    test_as_process()
    print("\ntest_as_thread:")
    test_as_thread()


if __name__ == "__main__":
    run()
