# standard library import
from time import time

class timer():
    """
    Class implementation for the Matlab functions tic and toc. 
    """

    def __init__(self):
        """
        Class initialization. 

        Attributes
        ----------
        tic : float
            starting time
        toc : float
            final time
        get_elapsed_time : float
            elapsed time between tic() and toc() calls in seconds
        """
        pass

    def start(self):
        """
        Begin timer.
        """
        self.tic = time()

    def stop(self):
        """
        End timer.
        """
        self.toc = time()
        self.get_elapsed_time = self.toc - self.tic
