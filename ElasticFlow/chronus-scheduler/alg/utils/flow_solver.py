from ortools.graph import pywarapgraph


class FlowSolver(object):
    def __init__(self, graph, name):
        self.graph = graph
        self.name = name
        self.rebuild_graph()
        assert name in ['MinCostFlow'], 'method {} is not supported'.format(name)


    def rebuild_graph(self, ):
        self.key2idx = dict()
        raise NotImplementedError
    

    def min_cost_flow_solver(self, ):
        solver = pywrapgraph.SimpleMinCostFlow()
        pass

    def solve(self, ):
        if name == 'MinCostFlow':
            return self.min_cost_flow_solver()
        else:
            raise NotImplementedError




