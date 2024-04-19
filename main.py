from __future__ import annotations
from typing import List, Set, Dict
from io import TextIOWrapper
import functools
import itertools

import sys
sys.setrecursionlimit(2500)

class Graph:
    def __init__(self, id: str) -> None:
        self.vertices = {}
        self.id = id

    def __str__(self) -> str:
        out = 'GRAPH ID: ' + self.id + '\n' 
        for vertex in self.vertices.values():
            out += vertex.__str__() + '\n'
        return out
    
    def __repr__(self) -> str:
        return self.__str__()

    def addVertex(self, lbl: str):
        self.vertices[lbl] = Vertex(lbl, self)

    def copy(self):
        graphCopy = Graph(self.id + 'c')
        for lbl, vertex in self.vertices.items():
            graphCopy.addVertex(vertex.label)
            for mapping in vertex.mappings:
                graphCopy.vertices[lbl].addMapping(mapping)
        for lbl, vertex in self.vertices.items():
            for neighbor in vertex.neighbors:
                graphCopy.vertices[lbl].addNeighbor(graphCopy.vertices[neighbor.label])
        return graphCopy

class Vertex:
    def __init__(self, label: str, graph: Graph) -> None:
        self.label = label
        self.graph = graph
        self.neighbors = set()
        self.mappings = set()

    def __str__(self) -> str:
        # + ' with neighbors: ' + str([(x.label + ' in graph ' + x.graph.id) for x in self.neighbors]) + ' and
        return self.label #+ ' in graph ' + self.graph.id #+ ' mappings ' + str([(x.label + ' in graph ' + x.graph.id) for x in self.mappings])
    
    def __repr__(self) -> str:
        return self.__str__()

    def addNeighbor(self, v: Vertex) -> None:
        self.neighbors.add(v)
    
    def addMapping(self, v: Vertex) -> None:
        self.mappings.add(v)

    def copy(self):
        return self.graph.copy().vertices[self.label]
    
    def fullGraph(self) -> str:
        return self.graph.__str__()

class Main:
    def __init__(self) -> None:
        self.id = 0
        self.success = 0
        self.failure = 0
        self.duplicates = 0
        self.found = {}
        self.firstOrder = []

    def run(self) -> None:
        fromDim = 4
        toDim = 3
        originalHC = self.emptyHypercube(fromDim)
        newHC = self.emptyHypercube(toDim)
        self.mapping({originalHC.vertices['0'*fromDim]: newHC.vertices['0'*toDim]}, toDim)
        # self.mapping3({},{originalHC.vertices['0'*fromDim]: newHC.vertices['0'*toDim].neighbors}, newHC.vertices['1'*toDim], originalHC.vertices['1'*fromDim])
        print(fromDim, '->', toDim, '| Successes:', self.success, ' Duplicates:', self.duplicates, ' Failures:', self.failure)

    def verifyMapping(self, mapping: Dict[Vertex, Vertex]):
        allVertices = mapping.keys()
        for v1 in allVertices:
            for v2 in allVertices:
                if("{0:b}".format(int(v1.label,2)^int(v2.label,2)).count('1') < "{0:b}".format(int(mapping[v1].label,2)^int(mapping[v2].label,2)).count('1')):
                    print('FAILED', mapping)
                    return

    def bfsVertex(self, v: Vertex) -> List[Vertex]:
        allVertices = [[v]]
        foundMoreNeighbors = True
        while foundMoreNeighbors:
            neighbors = set().union(*[v.neighbors for v in allVertices[-1]]).difference(set(allVertices[-2]) if len(allVertices) > 1 else set())
            if len(neighbors) < 1:
                foundMoreNeighbors = False
            else:
                allVertices.append(list(neighbors))
        return allVertices

    def mapping(self, fixedPoints: Dict[Vertex, Vertex], toDim: int):
        order = list(itertools.chain.from_iterable(self.bfsVertex(list(fixedPoints.keys())[0])))
        with open(f'C:/Users/thedi/Desktop/out.txt', 'w') as writer:
            self.mappingHelper({}, order, {f: t.neighbors for f, t in fixedPoints.items()}, writer, toDim)
    
    # fix two orthogonal axes in layer 1 of new cube
    def mappingHelper(self, mapping: Dict[Vertex, Vertex], order: List[Vertex], frontier: Dict[Vertex, Set[Vertex]], writer: TextIOWrapper, remainingDFToFix, firstPoint: bool = True) -> None:
        if len(order) == 0:
            keys = list(mapping.keys())
            keys.sort(key=str)
            sortedMapping = {i: mapping[i] for i in keys}
            if str(sortedMapping) in self.found.keys():
                self.duplicates += 1
            else:
                # self.verifyMapping(mapping)
                self.success += 1
                if (self.success) % 10000 == 0:
                    print(self.success, self.duplicates, self.failure)
                writer.write(str(mapping))
                writer.write('\n')
            return
        
        newOrder = order.copy()
        v = newOrder.pop(0)
        adjacencyList = frontier[v]

        candidates = functools.reduce(lambda accCandidates, adjacency: accCandidates.intersection(adjacency.neighbors.union({adjacency})), adjacencyList, list(adjacencyList)[0].neighbors.union({list(adjacencyList)[0]}))
        populatedVertices = set()
        if remainingDFToFix > 0 and not firstPoint:
            populatedVertices = set(mapping.values())
            candidates = candidates.intersection(populatedVertices).union({list(candidates.difference(populatedVertices))[0]} if len(candidates.difference(populatedVertices)) > 0 else set())
        
        if len(candidates) == 0:
            self.failure += 1
        
        for candidate in candidates:
            newFrontier = frontier.copy()
            newFrontier.pop(v)
            for neighbor in v.neighbors:
                if neighbor not in mapping.keys():
                    if neighbor not in newFrontier:
                        newFrontier[neighbor] = set()
                    newFrontier[neighbor] = newFrontier[neighbor].union({candidate})

            newMapping = mapping.copy()
            newMapping[v] = candidate
            self.mappingHelper(newMapping, newOrder, newFrontier, writer, remainingDFToFix - (0 if candidate in populatedVertices else 1), False)


    # Hypercube made of vertices without mappings
    def emptyHypercube(self, dim: int) -> Graph:
        graph = Graph(str(self.id))
        self.id += 1
        for i in range(2**dim):
            lbl = bin(i)[2:].zfill(dim)
            graph.addVertex(lbl)
        for lbl, vertex in graph.vertices.items():
            for i in range(dim):
                neighborlbl = lbl[:i] + ('1' if lbl[i] == '0' else '0') + lbl[i + 1:]
                vertex.addNeighbor(graph.vertices[neighborlbl])
        return graph   
        
Main().run()