from twisted.internet import defer
from twisted.internet.task import LoopingCall


class Observed():
    def __init__(self):

        self.data = {"state": -1}

        # self.deferred = defer.succeed(self._get_data(self.data))
        self.deferred = defer.Deferred()
        self.deferred.addCallback(lambda data: data)
        self.deferred.callback(self.data)

    def run(self, init_state=1):
        
        loop = LoopingCall(self.step, init_state)
        loop.start(0.3)
        self.reactor = loop.clock

        return self.reactor

    def step(self, state):
        '''This function should be overriden'''
        step = 3
        # print(state)
        self.set_data({"state": self.data["state"]+step})

    def set_data(self, data):
        self.deferred.addCallback(self._set_data, data)
        
        # self.reactor.callLater(0.01, d.callback, 3)
        return self.deferred

    def _set_data(self, old_data, new_data):
        self.data = new_data
        
        # we need return since it will be put into continuation chain
        return self.data

    def get_data(self):
        self.deferred.addCallback(self._get_data)
        
        return self.deferred

    def _get_data(self, data):
        print("from Observed._get_data")
        print(data)
        # ISSUE: Choose:
        # this function could just return self.data
        # there will be no problem with simultanioulsy
        # writing and reading from it since
        # it is all inside the continuation
        # but giving to it self.data instead of data
        # will remove errors when Reciver forgot to return data
        # in his con.addCallback called function!
        # v0:
        return data
        # v1:
        # return self.data
