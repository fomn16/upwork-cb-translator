from typing import Any, Callable, Generic, TypeVar
from threading import Thread
from queue import Queue, Empty
from abc import ABC, abstractmethod

PIPE_POOLING_TIMEOUT = 5

T_in = TypeVar('T_in') # type of input data
T_out = TypeVar('T_out') # type of output data

class BasePipeSection(ABC, Generic[T_in, T_out]):
    pipe_function: Callable[[T_in], T_out | None]
    buffer: Queue[T_in] = None
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

    def to(self, output_pipe:'BasePipeSection[T_out, Any]') -> 'BasePipeSection[T_out, Any]':
        """ connects the output of this pipe section to the input of the next """
        self.output_destination = output_pipe
        return output_pipe

    def send_to_next_pipe(self, output: T_out):
        if(self.output_destination != None and output != None):
            self.output_destination.receive(output)

    def loop_function(self) -> None:
        try:
            while self.is_open or not self.buffer.empty():
                received = self.unpack_buffer()
                if(received == None):
                    continue
                output = self.pipe_function(received)
                self.send_to_next_pipe(output)
        finally:
            if(self.output_destination != None):
                self.output_destination.close()

    def open(self) -> None:
        """ starts the pipeline processing loop """
        if(self.output_destination != None):
            self.output_destination.open()
        self.thread = Thread(target=self.loop_function, daemon=True)
        self.thread.start()
        
    def close(self) -> None:
        """ signals the pipe section must be closed when curreng processing is done """
        self.is_open = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join()


########## Class Implementations ##########

# the class used for transcription has its own internal queue,
# therefore this pipe implementation bypasses the pipe's queue calls
class TranscribePipe(BasePipeSection[bytes, str]):
    def aggregate_input(self, input:bytes, aggregate:bytes):
        return aggregate+input
    
    input_function: Callable[[bytes], None]
    pipe_function: Callable[[None], str | None]
    close_function: Callable[[None], str | None]
    def __init__(   self,
                    input_function: Callable[[bytes], None],
                    pipe_function: Callable[[None], str | None],
                    close_function: Callable[[None], str | None]):
        """
        pipe_function: A function that is ran in a loop in a separated thread:
            - receives: the input to the pipe section
            - returns: an output
        input_function: Function called on every input piece
        close_function: Function called before closing pipe. Output is passed onto next pipe before sendin the close signal foward
        """
        self.input_function = input_function
        self.pipe_function = pipe_function
        self.close_function = close_function

    def loop_function(self) -> None:
        try:
            while self.is_open:
                output = self.pipe_function()
                self.send_to_next_pipe(output)
            output = self.close_function()
            self.send_to_next_pipe(output)
        finally:
            if(self.output_destination != None):
                self.output_destination.close()

    def receive(self, data: bytes) -> None:
        self.input_function(data)

class TranslatePipe(BasePipeSection[str, str]):
    def aggregate_input(self, input:str, aggregate:str):
        return aggregate+input

# the class used for TTS streams the audio out, therefeore the loop function
# needs to be modified to send the output audio chunks to the next pipe as soon as possible
class TTSPipe(BasePipeSection[str, bytes]):
    def aggregate_input(self, input:str, aggregate:str):
        return aggregate+input
    
    pipe_function: Callable[[str, Callable[[bytes], None]], None]
    def __init__(   self,
                    pipe_function: Callable[[str, Callable[[bytes], None]], None]):
        """
        pipe_function: A function that is ran in a loop in a separated thread. it receives:
            - the input to the pipe section
            - the function it will call to send audio chunks to the next pipe
        """
        super().__init__(pipe_function)
        
    def loop_function(self) -> None:
        try:
            while self.is_open or not self.buffer.empty():
                received = self.unpack_buffer()
                if(received == None):
                    continue
                self.pipe_function(received, self.send_to_next_pipe)
        finally:
            if(self.output_destination != None):
                self.output_destination.close()

class AudioOutPipe(BasePipeSection[bytes, None]):
    def aggregate_input(self, input:bytes, aggregate:bytes):
        return aggregate+input
    
    def receive(self, data: bytes) -> None:
        self.pipe_function(data)