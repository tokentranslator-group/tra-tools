import time
from twisted.web import server

from tra_tools.logger.observed import Observed
from tra_tools.logger.sender import Sender


class Process():
    '''The progress for testing LogRunner'''
    def step(self):
        i = 0
        while True:
            i += 1 
            time.sleep(0.3)
            yield i


class LogRunner(Observed):
    '''For using this logger self.step method should be implemented:
    step: state
    '''
    def __init__(self, process, port=8081):
        # type: Iterable -> None
        Observed.__init__(self)

        self.done = False
        self.port = port
        self.process_gen = process.step()

    def step(self, *args, **kwargs):
        
        if not self.done:
            try:
                next_state = next(self.process_gen)
                
            except StopIteration:
                # TODO: or just return here?
                # or better - infinite loop until event
                # self.reactor.stop()
                # print("The logger server stopped")
                self.done = True
                print("Inference was done: Ctrl+C to stop logger")
                return

            # print(next_state)
        
            Observed.set_data(self, {"state": next_state})
        
    def run(self):
        print("Running at http://localhost:%d" % self.port)
        reactor = Observed.run(self)
        reactor.listenTCP(self.port, server.Site(Sender(self)))
        reactor.run()


    
