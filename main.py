from __future__ import annotations
import os
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
        out = "GRAPH ID: " + self.id + "\n"
        for vertex in self.vertices.values():
            out += vertex.__str__() + "\n"
        return out

    def __repr__(self) -> str:
        return self.__str__()

    def addVertex(self, lbl: str):
        self.vertices[lbl] = Vertex(lbl, self)

    def copy(self):
        graphCopy = Graph(self.id + "c")
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
        return (
            self.label
        )  # + ' in graph ' + self.graph.id #+ ' mappings ' + str([(x.label + ' in graph ' + x.graph.id) for x in self.mappings])

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
        self.fromDim = 3
        self.toDim = 4
        originalHC = self.emptyHypercube(self.fromDim)
        newHC = self.emptyHypercube(self.toDim)
        self.mapping(
            {originalHC.vertices["0" * self.fromDim]: newHC.vertices["0" * self.toDim]},
        )
        # print(fromDim, '->', toDim, '| Successes:', self.success, ' Duplicates:', self.duplicates, ' Failures:', self.failure)

    def verifyMapping(self, mapping: Dict[Vertex, Vertex]):
        allVertices = mapping.keys()
        for v1 in allVertices:
            for v2 in allVertices:
                if "{0:b}".format(int(v1.label, 2) ^ int(v2.label, 2)).count(
                    "1"
                ) < "{0:b}".format(
                    int(mapping[v1].label, 2) ^ int(mapping[v2].label, 2)
                ).count(
                    "1"
                ):
                    print("FAILED", mapping)
                    return

    def bfsVertex(self, v: Vertex) -> List[Vertex]:
        allVertices = [[v]]
        foundMoreNeighbors = True
        while foundMoreNeighbors:
            neighbors = (
                set()
                .union(*[v.neighbors for v in allVertices[-1]])
                .difference(set(allVertices[-2]) if len(allVertices) > 1 else set())
            )
            if len(neighbors) < 1:
                foundMoreNeighbors = False
            else:
                allVertices.append(list(neighbors))
        return allVertices

    # based on the idea that transforming the HC before constructing the mapping leads to duplicate inequalities
    def constructInequalities2(self, mapping) -> List[str]:
        self.generateLabels(None, None)

    def generateLabels(self, assigned, dfs):
        numLayers = self.fromDim
        for layer in range(assigned[-1][1] if len(assigned) > 0 else 1, numLayers + 1):
            pass

    # generates every possible spacing of labels in terms of their offset from the previous label
    # (eg. [1,2,1] indicates 'a' on row 1, 'b' on row 3, 'c' on row 4)
    # again *very* brute force
    def offsetsList(self, mapping) -> List[int]:
        out = []
        for numLabels in range(2, len(mapping)):
            offsets = [1] + [1 for x in range(numLabels - 1)]

            def iterateOffsets() -> bool:
                offsets[-1] += 1
                numOverflows = 0
                while sum(offsets) >= len(mapping):
                    numOverflows += 1
                    if numOverflows == numLabels:
                        return False
                    offsets[-numOverflows:] = [1 for x in offsets[-numOverflows:]]
                    offsets[-(numOverflows + 1)] += 1
                return True

            out += [offsets.copy()]
            while iterateOffsets():
                out += [offsets.copy()]
        return out

    # really really messy brute force way of calculating all possible inequalities until i can find patterns
    # starts by labeling the rows, then finding the corresponding column labels
    # row label sets that result in 1+ unlabeled columns are skipped
    # *most of this is just formatting the strings to look pretty*
    def constructInequalities(self, mapping, offsetsList=None) -> List[str]:
        inequalities = []
        if not offsetsList:
            offsetsList = self.offsetsList(mapping)

        lhsCols = [
            [int(str(k)[i]) for k in mapping.keys()] for i in range(self.fromDim)
        ]
        rhsCols = [
            [int(str(v)[i]) for v in mapping.values()] for i in range(self.toDim)
        ]

        for offset in offsetsList:
            indices = [sum(offset[: x + 1]) for x in range(len(offset))]
            rhsLabels = [
                sum([col[row] * 2**idx for idx, row in enumerate(indices)])
                for col in rhsCols
            ]
            lhsLabels = [
                sum([col[row] * 2**idx for idx, row in enumerate(indices)])
                for col in lhsCols
            ]
            if 0 not in lhsLabels and 0 not in rhsLabels:
                lhsterms = [
                    "S("
                    + ",".join(
                        [
                            f"{idx+1}"
                            for idx, char in enumerate(format(lbl, f"0{len(indices)}b"))
                            if char == "1"
                        ]
                    )
                    + ")"
                    for lbl in lhsLabels
                ]
                rhsterms = [
                    "S("
                    + ",".join(
                        [
                            f"{idx+1}"
                            for idx, char in enumerate(format(lbl, f"0{len(indices)}b"))
                            if char == "1"
                        ]
                    )
                    + ")"
                    for lbl in rhsLabels
                ]
                # remove identical terms that appear on both sides
                for term in lhsterms:
                    if term in rhsterms:
                        rhsterms.remove(term)
                        lhsterms.remove(term)
                inequality = (
                    str([format(idx, f"0{self.fromDim}b") for idx in indices])
                    + " --> "
                    + "+".join(lhsterms)
                    + " | "
                    + "+".join(rhsterms)
                )
                inequalities += [inequality]
        return inequalities

    # generates every possible mapping from a dim self.fromDim -> dim self.toDim hypercube
    # fixedPoints are just points we force to be in only one spot (ie. 0^fromDim always maps to 0^toDim)
    def mapping(self, fixedPoints: Dict[Vertex, Vertex]):
        order = list(
            itertools.chain.from_iterable(self.bfsVertex(list(fixedPoints.keys())[0]))
        )
        with open(f"{os.path.dirname(__file__)}\out.txt", "w") as writer:
            self.mappingHelper(
                {},
                order,
                {f: t.neighbors for f, t in fixedPoints.items()},
                writer,
                0b0,
                bin(2**self.toDim - 1),
            )

    # recursive DFS with a frontier that also keeps track of restrictions for *all* vertices any time a new vertex is mapped, to eliminate recalculation on each iteration
    def mappingHelper(
        self,
        mapping: Dict[Vertex, Vertex],
        order: List[Vertex],
        frontier: Dict[Vertex, Set[Vertex]],
        writer: TextIOWrapper,
        fixedDF: bin,  # binary string, 0 indicating an un-fixed DoF, and 1 indicating a fixed DoF
        targetFixedDF: bin,  # a string of 1s of length dim(RHS HC)
        firstPoint: bool = True,
    ) -> None:
        if len(order) == 0:
            keys = list(mapping.keys())
            keys.sort(key=str)
            sortedMapping = {i: mapping[i] for i in keys}
            if str(sortedMapping) in self.found.keys():
                self.duplicates += 1
            else:
                if int(str(mapping[keys[0]])) != 0:
                    return
                # self.verifyMapping(mapping)
                self.success += 1
                if (self.success) % 10000 == 0:
                    print(self.success, self.duplicates, self.failure)
                if int(targetFixedDF, 2) == functools.reduce(
                    lambda acc, x: acc | int(x.label, 2), mapping.values(), 0b0
                ):
                    toWriteList = list(mapping.items())
                    toWriteList.sort(key=lambda item: int(item[0].label, 2))
                    toWrite = ""
                    for item in toWriteList:
                        toWrite += f"{item[0]}: {item[1]} \n"
                    writer.write(str(toWrite))

                    # calculate all the inequalities -- this is super slow because it's brute force. if you just care about the mappings, comment this bit out
                    for inequality in self.constructInequalities(dict(toWriteList)):
                        writer.write(inequality)
                        writer.write("\n")
                    writer.write("\n")
                    # print("OK")
            return

        newOrder = order.copy()
        v = newOrder.pop(0)

        # frontier's been keeping track of who 'v' needs to be adjacent to
        adjacencyList = frontier[v]

        # this finds all of the nodes on the RHS HC that fit the adjacency criteria
        # this is efficient since we're saving references to persistent vertex objects that already keep track of their neighbors, rather than having to recalculate anything
        candidates = functools.reduce(
            lambda accCandidates, adjacency: accCandidates.intersection(
                adjacency.neighbors.union({adjacency})
            ),
            adjacencyList,
            list(adjacencyList)[0].neighbors.union({list(adjacencyList)[0]}),
        )

        # fix a # of linearly independent points (picture vector from 0^m to the point) equal to the dimension of the RHS HC
        # if you picture a mapping as a graph that's on the edges of the RHS HC, a new DoF amounts to adding an edge that's not parallel to any of the other edges in the graph
        if fixedDF != targetFixedDF and not firstPoint:
            noDFChangeCandidates = {
                candidate
                for candidate in candidates
                if int(candidate.label, 2) | fixedDF == fixedDF
            }
            newcandidates = noDFChangeCandidates.union(
                {list(candidates.difference(noDFChangeCandidates))[0]}
                if len(candidates) > len(noDFChangeCandidates)
                else set()
            )
            candidates = newcandidates

        if len(candidates) == 0:
            self.failure += 1

        for candidate in candidates:
            # update the frontier
            # this copy is a shallow copy, so we still hang on to the references to the same vertices
            newFrontier = frontier.copy()
            newFrontier.pop(v)
            for neighbor in v.neighbors:
                if neighbor not in mapping.keys():
                    if neighbor not in newFrontier:
                        newFrontier[neighbor] = set()
                    newFrontier[neighbor] = newFrontier[neighbor].union({candidate})

            newMapping = mapping.copy()
            newMapping[v] = candidate
            self.mappingHelper(
                newMapping,
                newOrder,
                newFrontier,
                writer,
                int(candidate.label, 2) | fixedDF,
                targetFixedDF,
                False,
            )

    # Hypercube made of vertices without mappings
    def emptyHypercube(self, dim: int) -> Graph:
        graph = Graph(str(self.id))
        self.id += 1
        for i in range(2**dim):
            lbl = bin(i)[2:].zfill(dim)
            graph.addVertex(lbl)
        for lbl, vertex in graph.vertices.items():
            for i in range(dim):
                neighborlbl = lbl[:i] + ("1" if lbl[i] == "0" else "0") + lbl[i + 1 :]
                vertex.addNeighbor(graph.vertices[neighborlbl])
        return graph


Main().run()
