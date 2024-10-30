from tra_tools.logger.server import CounterMix


class Sender(CounterMix):
    
    def _render_GET(self, request):

        con = self.observed.get_data()
        con.addCallback(self.step, request)
        print("observed params:")
        print(self.observed.data)
        print(self.observed.deferred)
        '''
        try:
            # TODO: move it on init:
            deferred.addCallback(self.step)
        except:
            pass
        '''

    # Qws; what if data never arrived?
    # Ans: if data never arrived it just return previous one
    # only problem will be if getting data is stack,
    # but it does not means that where no data send all will be frozen
    # since continuation just pass previous result farther in the chain
    # only make sure there is return data on each callback
    def step(self, data, request):
        print("Reciver: data recived from Sender:")
        print(data)
        request.write(str(data).encode())
        request.finish()

        # error should be triggered here:
        return data
        # but it just print
        # from Observed._get_data
        # None

