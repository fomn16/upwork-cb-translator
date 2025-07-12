from typing import Any, Callable, Generic, TypeVar
from threading import Thread
from queue import Queue, Empty
from abc import ABC, abstractmethod

PIPE_POOLING_TIMEOUT = 5

T_in = TypeVar('T_in') # type of input data
T_out = TypeVar('T_out') # type of output data

class BasePipeSection(ABC, Generic[T_in, T_out]):
    pipe_function: Callable[[T_in], T_out]
    buffer: Queue[T_in]
    is_open: bool = True

    output_destination: 'BasePipeSection[T_out, Any]' = None
    thread: Thread = None

    def __init__(   self,
                    pipe_function: Callable[[T_in], T_out]):
        """
        pipe_function: A function that is ran in a loop in a separated thread:
            - receives: the input to the pipe section
            - returns: an output
        """
        self.pipe_function = pipe_function
        self.buffer = Queue()
            
    @abstractmethod
    def aggregate_input(self, input:T_in, aggregate:T_in) -> T_in:
        """
        function that 
            - takes a piece of data from input
            - aggregates it (appends it) to aggregate
            - (optionally) preprocesses contents
            - returns aggregated results
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def unpack_buffer(self)->T_in:
        try:
            unpacked = self.buffer.get(timeout=PIPE_POOLING_TIMEOUT)
        except Empty:
            return None
        
        while not self.buffer.empty():
            unpacked = self.aggregate_input(self.buffer.get(), unpacked)
        return unpacked

    def receive(self, data: T_in) -> None:
        self.buffer.put(data)

    def output_to(self, output_pipe:'BasePipeSection[T_out, Any]') -> None:
        """ connects the output of this pipe section to the input of the next """
        self.output_destination = output_pipe

    def loop_function(self) -> None:
        try:
            while self.is_open or not self.buffer.empty():
                received = self.unpack_buffer()
                if(received == None):
                    continue
                output = self.pipe_function(received)
                if(self.output_destination != None):
                    self.output_destination.receive(output)
        finally:
            if(self.output_destination != None):
                self.output_destination.close()

    def open(self) -> None:
        """ starts the pipeline processing loop """
        self.thread = Thread(target=self.loop_function, daemon=True)
        self.thread.start()
        
    def close(self) -> None:
        """ signals the pipe section must be closed when curreng processing is done """
        self.is_open = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join()

class StringToStringPipe(BasePipeSection[str, str]):
    def aggregate_input(self, input:str, aggregate:str):
        return aggregate+input
    
class StringEndOfPipe(BasePipeSection[str, None]):
    def aggregate_input(self, input:str, aggregate:str):
        return aggregate+input