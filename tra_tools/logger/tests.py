from twisted.web import server

from tra_tools.logger.logger import LogRunner, Process
from tra_tools.logger.observed import Observed
from tra_tools.logger.sender import Sender


def test1():
    print("Running test 1")
    runner = LogRunner(Process())
    runner.run()


def test0():

    print("Running test 0 at http://localhost:8081")
    observed = Observed()
    
    reactor = observed.run()

    reactor.listenTCP(8081, server.Site(Sender(observed)))
    reactor.run()


if __name__ == "__main__":
    test1()
    # test0()
