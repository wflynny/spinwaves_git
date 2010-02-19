
'''
This module is developed by University of Maryland as part of the Distributed
Data Analysis of Neutron Scattering Experiments (DANSE) project funded by the
US National Science Foundation.

Please read license.txt for module usage

copyright 2009, University of Maryland for the DANSE project


A Mapper is consists of a set of worker processes for performing Parallel map
requests. Mapper provides a way to submit jobs (map requests), and get the
results back when they are available. The Mapper is responsible for assigning
jobs to the worker processes by putting them in a input queue, where they are
picked up by the next available worker. The worker then performs the assigned 
map request in the background and puts the processed request in an output queue.

Currently Mapper uses Python Multiprocessing 'Process' module as worker
'''

try:
    import multiprocessing
except ImportError:
    print ' Multiprocessing is not available'

# Misc. imports
import sys
import Queue
import time
import logging
import traceback


# set-up logging
#logger = logging.getLogger('Mapper')

# Item pushed on the input queue to tell the worker process to terminate
SENTINEL = "QUIT"


def is_sentinel(obj):
    """
    Predicate to determine whether an item from the queue is the signal to stop.
    """
    return type(obj) is str and obj == SENTINEL

class Mapper(object):
    """
    Mapper class, distributes map requests and collectes them after they are
    processed. It used Python's built-in 'map' like style and evalutes function 
    on each elements of an iterable.
    """

    def __init__(self,  num_workers=0,input_queue_size=0, output_queue_size=0):
        """
        Set up the mapper and start num_workers worker processes.

        *num_workers* The number of worker processes to start initially.

        .. note::
            Currently defualt num_workers is total number of avialble CPUs.

        *input_queue_size* If a positive integer, it's the maximum number of
         unassigned jobs. The mapper blocks when the queue is full and a new job
         is submitted.
        *output_queue_size* If a positive integer, it's the maximum number of
         completed jobs waiting to be fetched. The mapper blocks when the queue
         is full and a job is completed.
        """

        self.closed = False
        self.workers = []
        self.activekey2job = {}
        self.unassignkey2job = {}
        self.unassigned_jobs = multiprocessing.Queue(input_queue_size)
        self.processed_jobs = multiprocessing.Queue(output_queue_size)
        self.num_workers = multiprocessing.cpu_count()
        self.add_workers(self.num_workers)
        self.outlist = []
        
    def map(self, f, v):
        """
        A parallel equivalent of Python's built-in 'map()' function. It blocks 
        till the result is ready.

        *f* name of the function
        *v* iterable
        """
        self.func = f
        self.data = v
        
        self.p = self.data[0]
        self.q = self.data[1]
        self.r = self.data[2]
        self.arg = self.data[3]
        
        if len(self.workers)>0:
            chunksize, extra = divmod(len(self.p), len(self.workers)*4)
            if extra:
                chunksize += 1
        else:
            print ('No worker available')
            pass
        
        argindex = 0
        self.outlist = [None]*len(self.p) 
        
        iter_element = iter(self.p)
        exit_loop = False
        chunk = []
        while not exit_loop:
            seq = []
            for i in xrange(chunksize or 1):
                try:
                    args = iter_element.next()
                    #print 'arg', args
                except StopIteration:
                    exit_loop = True
                    break
                job = Map_job(self.func, argindex,args,self.q[0],self.r,self.arg)
                argindex += 1
                seq.append(job)
                
            chunk.append(Map_chunk(seq))

        for seq in chunk:
            self.add_mapjob(seq)
        
        result = self.collect_work()
        return result
    
    def add_workers(self, n):
        """
        Add worker processes to the pool.
        """
        for _ in xrange(n):
            self.workers.append(Worker(self.unassigned_jobs, self.processed_jobs))
                               
    def add_mapjob(self, job):
        """
        Add a map-request to the end of input queue.
        
        *timeout* If the input queue is full and timeout is None, block until a
        slot becomes avilable. If timeout >0, block for uptp timeout seconds and
        raise Queue.Full exception if the queue is still full. If timeout <=0, 
        do not block and raise Queue.Full immediately if the queue is full.
        """
        key = job.key
        self.unassigned_jobs.put(job)
        self.unassignkey2job[key] = self.activekey2job[key] = job
              
    
    def iter_processed_jobs(self):
        # Returns an iterator over the finished mapjobs, popping them off from 
        # the processed_jobs queue                
       
        while self.activekey2job:
            try: 
                job = self.processed_jobs.get()
            except Queue.Empty:
                logger.debug('queue empty')
                break
            key = job.key
            
            # at this point the key is guaranteed to be in activekey2job even
            # if the job has been cancelled
            assert key in self.activekey2job
            del self.activekey2job[key]
            yield job

    def collect_work(self):
        # Return a list of finished mapjob requests.
        try:
            while len(self.outlist)>=0:
                res = self.iter_processed_jobs().next()
                result = res.result()
                for i in result:
                    returnindex =  i[0]
                    value = i[1]
                    self.outlist[returnindex] = value  
                    
        except StopIteration:
            pass
            
        return self.outlist

    def num_of_worker(self):
        return len(self.workers)

    def num_active_jobs(self):
        # Return the approimate number of active mapjobs
        return len(self.activekey2job)     
  
    def kill_worker(self, n=1):
        """
        Tell the worker process to quit after they finish thier current map-request.
        """
        for _ in xrange(n):
            try:
                self.workers.pop().dismissed = True
            except KeyError:
                break   
    
    def close(self):
        """
        Prevents any more map-requests from being submitted to pool. Once all
        the requests have been completed the worker process will exit.
        """
        self.closed = True

    def terminate(self):
        """
        Stops the worker processes immediately without completing outstanding
        work. When the pool object is garbage collected terminate() will be
        called immediately.
        """
        self.close()
        # Clearing the job queue
        try:
            while 1:
                self.unassigned_jobs.get_nowait()
        except Queue.Empty:
            pass

        # Send one sentinel for each worker process: each process will die
        # eventually, leaving the next sentinel for the next process
        for worker in self.workers:
            self.unassigned_jobs.put(SENTINEL) 
    
    def join(self):
        """Wait for the worker processes to exit. One must call close() or
        terminate() before using join()."""
        for worker in self.workers:
            worker.join()
                      
 
#==========================Map job class========================================
       
class WorkUnit(object):
    """
    Base class for a unit of work submitted to the worker process. It's
    basically just an object with a process() and result() method. This class is
    not directly usable.
    """
    def process(self):
        """
        Do the work. 
        """
        raise NotImplementedError("Children must override Process")
  
    def result(self):
        """
        Store the result of work
        """
        raise NotImplementedError("Children must override Process")


class Map_job(WorkUnit):
    """
    A work unit that corresponds to the execution of a map request. A maprequest
    executes a callable and send its result or exception information.
    """
    
    def __init__(self, func, index, datax, datay, dataz, datarg):
        self.func = func
        self.index = index
        self.x = datax
        self.y = datay
        self.z = dataz
        self.arg = datarg
        self.map_key = id(self)

    def process(self):
        """
        process the actual work with given arguments and store the result or its
        exception information.
        """
        try:
            self._result = self.func(self.x,self.y,self.z,*self.arg)
            #print 'result', self._result
        except:
            self._exc_info = traceback.format_exc()
        else:
            self._exc_info = None
                
#        try:
#            if isinstance(self.data, list) or isinstance(self.data, tuple):
#                self._result = self.func(self.x,self.y,*self.arg)
#                print 'result', self._result
#            elif isinstance(self.data, dict):
#                self._result = self.func(**self.data)
#        except:
#            self._exc_info = traceback.format_exc()
#        else:
#            self._exc_info = None
                                
    def result(self):
        """
        Store the result of map_job or its exception information.       
        """
        if self._exc_info is not None:
            t = self._exc_info
            print('Service has error %s: '%t)
            
        try:
            print [self.index, self._result]
            return [self.index, self._result]
        except AttributeError:
            raise
 
    
class Map_chunk(WorkUnit):
    """
    Class that corresponds to the processing of a continuous sequence of 
    Map_job objects
    """
    def __init__(self, jobs):
        WorkUnit.__init__(self)
        self._jobs = jobs
        self.key = id(self)

    def process(self):
        """
        Call process() on all the Map_job objects that have been specified
        """
        for job in self._jobs:
            job.process()
            
    def result(self):
        """
        Store the result of chunk of map_job
        """
        a =[]
        for job in self._jobs:
            p = job.result()
            a.append(p)
        return a

#===========================Worker class=======================================

class Worker(multiprocessing.Process):
    """   
    Background process connected to the input/output job request queues.
    A worker process sits in the background and picks up map_job requests from
    one queue and puts the processed requests in another, until it is killed.
    """

    def __init__(self, inputQueue,  outputQueue, **kwds):
        """
        Set up worker process in daemonic mode and start it immediatedly.
        class when it creates a new worker process.
        """
        super(Worker,self).__init__(**kwds)
        self.daemon =True
        self.inputQueue = inputQueue
        self.outputQueue = outputQueue
        self.dismissed = False
        self.start()
        
    def run(self):
        """
        Poll the input job queue indefinitely or until told to exit. Once a job
        request has been popped from the input queue, process it and add it to
        the output queue if it's not cancelled, otherwise discard it.
        """
        
        while True:
            # process blocks here if inputQueue is empty
            try:
                job = self.inputQueue.get()
            except UnboundLocalError:
                logger.error("problem in polling map job from inputQueue")
            if is_sentinel(job):
                break            
            key = job.key
                        
            if self.dismissed: # put back the job we just picked up and exit
                self.inputQueue.put(job)
                break
            
            job.process()
            
            # process blocks here if outputQueue is full
            self.outputQueue.put(job)
            
def func(y,x,z,*args):
    print 'x',x
    print 'y',y
    print 'z',z
    return x*y        

if __name__=='__main__':
   
    x = [1,2,3]
    y = [4,5,6]
    z = [7,8,9]
    arg = ['alpha','beta',7,8]
    pool = Mapper()
    result = pool.map(func, (x,y,z,arg))
    print 'result', result
    pool.terminate()
    pool.join()
    
