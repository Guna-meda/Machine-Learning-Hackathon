"""Road-network helpers for the HMM: arc geometry, arc-to-arc distances, route stitching.

Everything is per trip (each trip has its own network).  Distances are in projected metres.
"""
import heapq
import numpy as np


class RoadNetwork:
    def __init__(self, trip):
        self.arcs = trip.arcs
        self.xy = trip.nodes_xy
        self.tail = trip.arcs[:, 0]           # node where the arc starts
        self.head = trip.arcs[:, 1]           # node where the arc ends
        d = self.xy[self.head] - self.xy[self.tail]
        self.length = np.hypot(d[:, 0], d[:, 1])
        # outgoing arcs per node (built lazily as a dict of lists, only for nodes we touch)
        order = np.argsort(self.tail, kind="stable")
        self._sorted_arcs = order
        self._start = np.searchsorted(self.tail[order], np.arange(len(self.xy) + 1))
        self._reach_cache = {}

    def out_arcs(self, node):
        return self._sorted_arcs[self._start[node]:self._start[node + 1]]

    def reach(self, node, bound):
        """Shortest driving distance from `node` to every node within `bound` metres.

        Returns (dist, via) dicts; via[n] = (previous node, arc used) lets us rebuild the path.
        Results are cached because the same nodes are asked about again and again.
        """
        key = (node, bound)
        if key in self._reach_cache:
            return self._reach_cache[key]
        dist, via = {node: 0.0}, {}
        heap = [(0.0, node)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf):
                continue
            for a in self.out_arcs(u):
                v = self.head[a]
                nd = d + self.length[a]
                if nd <= bound and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    via[v] = (u, a)
                    heapq.heappush(heap, (nd, v))
        if len(self._reach_cache) > 200000:
            self._reach_cache.clear()
        self._reach_cache[key] = (dist, via)
        return dist, via

    def path_arcs(self, from_arc, to_arc, bound):
        """Arcs that must be driven between two (different) arcs, or None if not reachable within bound."""
        start, goal = self.head[from_arc], self.tail[to_arc]
        if start == goal:
            return []
        dist, via = self.reach(start, bound)
        if goal not in dist:
            return None
        arcs, node = [], goal
        while node != start:
            prev, a = via[node]
            arcs.append(int(a))
            node = prev
        return arcs[::-1]


def stitch_route(net, arc_sequence, bound=1500.0):
    """Turn per-point arc predictions into one connected route (list of arc ids).

    Consecutive duplicates are collapsed.  If two consecutive arcs are not joined, the
    shortest path between them is inserted (up to `bound` metres).  If no such path exists
    the route is left with a break and the break position is reported.
    """
    route, breaks = [], []
    for a in arc_sequence:
        a = int(a)
        if a < 0:
            continue
        if not route:
            route.append(a)
            continue
        if a == route[-1]:
            continue
        gap = net.path_arcs(route[-1], a, bound)
        if gap is None:
            breaks.append(len(route))
        else:
            route.extend(gap)
        route.append(a)
    return route, breaks
