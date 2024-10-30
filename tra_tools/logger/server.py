from twisted.web import server, resource
from twisted.web.resource import Resource


class CounterMix(Resource):
    '''Basic class for two logger servers'''
    
    isLeaf = True  # this is important, otherwise nothing will work

    counter = 0
    
    def __init__(self, observed, *args, **kwargs):

        # Resource.__init__(self)
        resource.Resource.__init__(self, *args, **kwargs)
        self.observed = observed
    
    def render_GET(self, request):
        # REF: https://github.com/ktuite/twisted-cors/blob/master/mainServer.py
        # these are the CROSS-ORIGIN RESOURCE SHARING headers required
        # learned from here: http://msoulier.wordpress.com/2010/06/05/cross-origin-requests-in-twisted/
        request.setHeader('Access-Control-Allow-Origin', '*')
        request.setHeader('Access-Control-Allow-Methods', 'GET')
        request.setHeader('Access-Control-Allow-Headers', 'x-prototype-version,x-requested-with')
        request.setHeader('Access-Control-Max-Age', '2520') # 42 hours
    
        # normal JSON header
        request.setHeader('Content-type', 'application/json')
        self.counter += 1
        self._render_GET(request)

        # signal that the rendering is not complete
        return server.NOT_DONE_YET
        
    def _render_GET(self, request):
        request.finish()
