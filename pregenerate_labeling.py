import itertools
import time
import numpy as np

globalFound = 0
globalEliminated = 0


def getAllNumsStratified(numDigits):
    allNums = [bin(n)[2:].zfill(numDigits) for n in reversed(range(1, 2**numDigits))]
    return [
        [
            [int(digit) for digit in bin]
            for bin in filter(lambda num: num.count("1") == n, allNums)
        ]
        for n in range(1, numDigits + 1)
    ]


def permutationMatrices(numCols):
    numCols
    permutations = list(itertools.permutations(range(numCols)))

    matrices = []
    for perm in permutations:
        matrices.append(
            np.transpose(
                np.matrix(
                    [[1 if n == col else 0 for n in range(numCols)] for col in perm]
                )
            )
        )
    return matrices


def getPermutationHash(perm, mat):
    out = "".join(
        map(
            str,
            itertools.chain.from_iterable(
                sorted(
                    np.matmul(mat, perm).tolist(),
                    key=lambda x: (sum(x), -int("".join(map(str, x)))),
                )
            ),
        )
    )
    return out


def subsequentLabels(currentLabels, remainingCandidates, permMats, cap=None):
    global globalFound
    if cap:
        if len(currentLabels) == cap:
            globalFound += 1
            if globalFound % 1000 == 0:
                print(globalFound)
            return [currentLabels]

    if len(remainingCandidates) == 0:
        globalFound += 1
        if globalFound % 1000 == 0:
            print(globalFound)
        return [currentLabels]

    stratefiedCandidates = [
        list(filter(lambda cand: sum(cand[1]) == n, enumerate(remainingCandidates)))
        for n in range(sum(currentLabels[-1]), sum(remainingCandidates[-1]) + 1)
    ]
    validPerms = permMats.copy()

    stratefiedCurrentLabels = [
        list(filter(lambda cand: sum(cand) == n, currentLabels))
        for n in range(sum(currentLabels[-1]) + 1)
    ]

    for strata in stratefiedCurrentLabels:
        if len(strata) == 0:
            continue
        strataMatrix = np.matrix(strata)
        strataHash = "".join(map(str, strataMatrix.flat))
        validPerms = list(
            filter(
                lambda perm: getPermutationHash(perm, strataMatrix) == strataHash,
                validPerms,
            )
        )

    # currLabelMatrix = np.matrix(currentLabels)
    # currLabelHash = "".join(map(str, currLabelMatrix.flat))

    # validPerms = list(
    #     filter(
    #         lambda perm: getPermutationHash(perm, currLabelMatrix) == currLabelHash,
    #         permMats,
    #     )
    # )

    indepCands = []
    for strata in stratefiedCandidates:
        cands = strata.copy()
        seen = set()
        for candAndIdx in cands:
            idx, cand = candAndIdx
            if "".join(map(str, cand)) in seen:
                continue
            # mat = np.vstack([currLabelMatrix, np.matrix(cand)])
            results = map(
                lambda perm: getPermutationHash(perm, np.matrix(cand)), validPerms
            )
            seen.update(results)
            indepCands.append((cand, idx))

    allLabelSequences = [currentLabels]
    globalFound += 1
    if globalFound % 1000 == 0:
        print(globalFound)

    for cand, idx in indepCands:
        allLabelSequences += subsequentLabels(
            currentLabels + [cand], remainingCandidates[idx + 1 :], permMats, cap
        )
    return allLabelSequences


def getAllSequences(dim, lengthCap=None):
    allLabels = getAllNumsStratified(dim)
    flattened = list(itertools.chain.from_iterable(allLabels))
    startingPoints = [strata[0] for strata in allLabels]

    allSequences = {}
    for startingPoint in startingPoints:
        allSequences["".join(map(str, startingPoint))] = subsequentLabels(
            [startingPoint],
            flattened[flattened.index(startingPoint) + 1 :],
            permutationMatrices(dim),
            lengthCap,
        )
    return allSequences


start = time.time()

allSequences = getAllSequences(dim=7, lengthCap=6)
numSeqs = 0
for k, v in allSequences.items():
    numSeqs += len(v)
    # print(k)
    # for seq in v:
    #     print(["".join(map(str, point)) for point in seq])
    # print()

print(numSeqs)

fin = time.time()

print("Took", fin - start)
