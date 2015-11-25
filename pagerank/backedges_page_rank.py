from simple_page_rank import SimplePageRank

"""
This class implements the pagerank algorithm with
backwards edges as described in the second part of 
the project.
"""
class BackedgesPageRank(SimplePageRank):

    """
    The implementation of __init__ and compute_pagerank should 
    still be the same as SimplePageRank.
    You are free to override them if you so desire, but the signatures
    must remain the same.
    """

    """
    This time you will be responsible for implementing the initialization
    as well. 
    Think about what additional information your data structure needs 
    compared to the old case to compute weight transfers from pressing
    the 'back' button.
    """
    @staticmethod
    def initialize_nodes(input_rdd):
        # takes in a line and emits edges in the graph corresponding to that line
        def emit_edges(line):
            # ignore blank lines and comments
            if len(line) == 0 or line[0] == "#":
                return []
            # get the source and target labels
            source, target = tuple(map(int, line.split()))
            # emit the edge
            edge = (source, frozenset([target]))
            # also emit "empty" edges to catch nodes that do not have any
            # other node leading into them, but we still want in our list of nodes
            self_source = (source, frozenset())
            self_target = (target, frozenset())
            return [edge, self_source, self_target]

        # collects all outgoing target nodes for a given source node
        def reduce_edges(e1, e2):
            return e1 | e2 

        # sets the weight of every node to 0, and formats the output to the 
        # specified format of (source (old_weight, weight, targets))
        def initialize_weights((source, targets)):
            return (source, (1.0, 1.0, targets))

        nodes = input_rdd\
                .flatMap(emit_edges)\
                .reduceByKey(reduce_edges)\
                .map(initialize_weights)
        return nodes

    """
    You will also implement update_weights and format_output from scratch.
    You may find the distribute and collect pattern from SimplePageRank
    to be suitable, but you are free to do whatever you want as long
    as it results in the correct output.
    """
    @staticmethod
    def update_weights(nodes, num_nodes):
        """
        Mapper phase.
        Distributes pagerank scores for a given node to each of its targets,
        as specified by the update algorithm.
        Some important things to consider:
        We can't just emit (target, weight) values to the reduce phase, 
        because then the reduce phase will lose information on the outgoing edges
        for the nodes. We have to emit the (node, targets) pairs too so that 
        the edges can be remembered for the next iteration.
        Think about the best output format for the mapper so the reducer can
        get both types of information.
        You are allowed to change the signature if you desire to.
        """
        def distribute_weights((node, (old_weight, weight, targets))):
            p = list()
            m = (node, (weight, (0.05*weight) + (old_weight * 0.1), targets))
            p.append(m)
            if type(targets) is frozenset:
                num = len(targets)
                if num != 0:
                    for t in targets:
                        k = (0.85/num) * weight
                        p.append((t, (0.0, k, [])))
                else:
                    for i in range(0, num_nodes):
                        if i != node:
                            p.append((i, (0.0, (0.85/(num_nodes-1))*weight, [])))
            return p

        """
        Reducer phase.
        We are given a node as a key and a list of all the values emitted by the mappers
        corresponding to that key.
        There should be two types of values:
        Pagerank scores, which represent how much score an incoming node is giving to us,
        and edge data, which we need to collect and store for the next iteration.
        The output of this phase should be in the same format as the input to the mapper.
        You are allowed to change the signature if you desire to.
        """
        def collect_weights((node, values)):   
            sumw = 0.0 
            tar = []
            for v in values:
                i = 0
                while i <= 2:
                    if type(v[i]) is frozenset: 
                        tar = v[i]
                    elif type(v[i]) is float or type(v[i]) is int:
                        if i == 0:
                            if v[0] != 0.0:
                                old = v[0]
                        else:
                            sumw += v[i]
                    i += 1
            return (node, (old, sumw, tar))

        return nodes\
                .flatMap(distribute_weights)\
                .groupByKey()\
                .map(collect_weights)

    """
    Formats the output of the data to the format required by the specs.
    If you changed the format of the update_weights method you will 
    have to change this as well.
    Otherwise, this is fine as is.
    """
    @staticmethod
    def format_output(nodes):
        return nodes\
                .map(lambda (node, (old_weight, weight, targets)): (weight, node))\
                .sortByKey(ascending = False)\
                .map(lambda (weight, node): (node, weight))

