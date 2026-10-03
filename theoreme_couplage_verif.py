#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theoreme_couplage_verif.py

Vérification computationnelle du Théorème A :
    rang(M_{p,d,r}(sigma,t)) <= nu(G_{p,d,r})

Le cœur mathématique est celui du document de référence :
    f(x) = x^d + sigma*x^r + t
    n = (p-1)//2
    g = floor((d-1)/2)

T(i,j) = {(alpha,beta,gamma) in N^3 :
          alpha+beta+gamma=n,
          d*alpha+r*beta = i*p-j}

Une arête (i,j) existe exactement lorsque T(i,j) est non vide.
Le programme construit G, calcule un couplage maximum par l'algorithme
de Kuhn, construit la matrice de Cartier--Manin et calcule son rang
exact sur F_p.

Aucune égalité rang = nu n'est supposée.
"""

from math import gcd, comb
from typing import List, Tuple, Optional

Edge = Tuple[int, int]


def graph_edges(p: int, d: int, r: int) -> List[Edge]:
    """Construit les arêtes du graphe G_{p,d,r}."""
    if p <= 2 or p % 2 == 0:
        raise ValueError("p doit être un nombre premier impair.")
    if d < 3:
        raise ValueError("d >= 3.")
    if not (1 <= r <= d - 1):
        raise ValueError("1 <= r <= d-1.")

    n = (p - 1) // 2
    g = (d - 1) // 2
    edges = []

    for i in range(1, g + 1):
        for j in range(1, g + 1):
            target = i * p - j
            found = False

            # On parcourt beta. Alors alpha est imposé.
            for beta in range(n + 1):
                rem = target - r * beta
                if rem < 0:
                    continue
                if rem % d != 0:
                    continue
                alpha = rem // d
                gamma = n - alpha - beta
                if alpha >= 0 and gamma >= 0:
                    found = True
                    break

            if found:
                edges.append((i, j))

    return edges


def graph_adjacency(p: int, d: int, r: int):
    """Adjacence gauche -> droite, avec sommets numérotés 1..g."""
    g = (d - 1) // 2
    adj = {i: [] for i in range(1, g + 1)}
    for i, j in graph_edges(p, d, r):
        adj[i].append(j)
    return adj


def maximum_matching(p: int, d: int, r: int):
    """
    Couplage maximum par algorithme de Kuhn.
    Retourne (nu, match_left, match_right).
    """
    g = (d - 1) // 2
    adj = graph_adjacency(p, d, r)

    match_right = {j: None for j in range(1, g + 1)}

    def dfs(i, seen):
        for j in adj[i]:
            if j in seen:
                continue
            seen.add(j)
            if match_right[j] is None or dfs(match_right[j], seen):
                match_right[j] = i
                return True
        return False

    nu = 0
    for i in range(1, g + 1):
        if dfs(i, set()):
            nu += 1

    match_left = {i: None for i in range(1, g + 1)}
    for j, i in match_right.items():
        if i is not None:
            match_left[i] = j

    return nu, match_left, match_right


def matrix_symbolic_data(p: int, d: int, r: int):
    """
    Retourne la liste T(i,j), donc les monômes qui composent chaque entrée.

    Chaque élément est (alpha,beta,gamma), coefficient multinomial.
    """
    n = (p - 1) // 2
    g = (d - 1) // 2
    data = {}

    for i in range(1, g + 1):
        for j in range(1, g + 1):
            target = i * p - j
            triples = []

            for beta in range(n + 1):
                rem = target - r * beta
                if rem < 0 or rem % d != 0:
                    continue
                alpha = rem // d
                gamma = n - alpha - beta
                if alpha < 0 or gamma < 0:
                    continue

                coeff = comb(n, alpha) * comb(n - alpha, beta)
                triples.append((alpha, beta, gamma, coeff % p))

            data[(i, j)] = triples

    return data


def matrix_mod_p(p: int, d: int, r: int, sigma: int, t: int):
    """Construit M modulo p pour des valeurs sigma,t dans F_p."""
    n = (p - 1) // 2
    g = (d - 1) // 2
    data = matrix_symbolic_data(p, d, r)

    M = [[0 for _ in range(g)] for _ in range(g)]

    for i in range(1, g + 1):
        for j in range(1, g + 1):
            value = 0
            for alpha, beta, gamma, coeff in data[(i, j)]:
                value += coeff * pow(sigma % p, beta, p) * pow(t % p, gamma, p)
            M[i - 1][j - 1] = value % p

    return M


def rank_mod_p(A: List[List[int]], p: int) -> int:
    """Rang exact d'une matrice sur F_p par élimination de Gauss."""
    if not A:
        return 0

    A = [row[:] for row in A]
    m = len(A)
    n = len(A[0]) if m else 0
    rank = 0

    for col in range(n):
        pivot = None
        for row in range(rank, m):
            if A[row][col] % p:
                pivot = row
                break

        if pivot is None:
            continue

        A[rank], A[pivot] = A[pivot], A[rank]
        inv = pow(A[rank][col] % p, -1, p)
        A[rank] = [(x * inv) % p for x in A[rank]]

        for row in range(m):
            if row != rank and A[row][col] % p:
                factor = A[row][col] % p
                A[row] = [
                    (A[row][k] - factor * A[rank][k]) % p
                    for k in range(n)
                ]

        rank += 1
        if rank == m:
            break

    return rank


def verify_instance(p: int, d: int, r: int,
                    sigma: int = 1, t: int = 1, verbose: bool = True):
    """Vérifie une instance numérique sans supposer l'égalité rang = nu."""
    g = (d - 1) // 2
    edges = graph_edges(p, d, r)
    nu, match_left, match_right = maximum_matching(p, d, r)
    M = matrix_mod_p(p, d, r, sigma, t)
    rank = rank_mod_p(M, p)

    result = {
        "p": p, "d": d, "r": r, "g": g,
        "edges": edges, "nu": nu,
        "sigma": sigma % p, "t": t % p,
        "matrix": M, "rank": rank,
        "bound_verified": rank <= nu,
        "non_ordinarity_for_this_instance": rank < g
    }

    if verbose:
        print(f"p={p}, d={d}, r={r}, g={g}")
        print(f"edges = {edges}")
        print(f"nu(G) = {nu}")
        print(f"M({sigma},{t}) mod {p} =")
        for row in M:
            print(row)
        print(f"rank(M) = {rank}")
        print(f"rank <= nu : {rank <= nu}")
        print(f"rank < g   : {rank < g}")
        print()

    return result


def test_theorem_B(p: int):
    """
    Tranche :
        p ≡ 5 (mod 6),
        d = p+1,
        r = (p+1)/3.

    Le document démontre :
        nu(G) = r = (p+1)/3,
        g - nu = (p-5)/6.
    """
    if p % 6 != 5:
        raise ValueError("Pour le Théorème B, il faut p ≡ 5 (mod 6).")

    d = p + 1
    r = d // 3
    g = (d - 1) // 2
    nu, _, _ = maximum_matching(p, d, r)

    expected_nu = r
    expected_defect = (p - 5) // 6

    print(f"p={p}, d={d}, r={r}, g={g}")
    print(f"nu(G) = {nu}, attendu = {expected_nu}")
    print(f"g-nu = {g-nu}, attendu = {expected_defect}")
    print(f"Théorème B vérifié numériquement : "
          f"{nu == expected_nu and g - nu == expected_defect}")

    return nu == expected_nu and g - nu == expected_defect


def demo():
    # Exemples explicitement présents dans le document de référence.
    print("=== Exemple 1 ===")
    verify_instance(11, 12, 4, 1, 1)

    print("=== Exemple 2 ===")
    verify_instance(7, 12, 3, 1, 1)

    print("=== Tranche de Sofyen : d=7, r=1 ===")
    for p in [3, 7, 11, 13, 17, 19, 23]:
        nu, _, _ = maximum_matching(p, 7, 1)
        print(f"p={p}: nu={nu}, g=3, nu<g={nu<3}")

    print("\n=== Théorème B ===")
    for p in [11, 17, 23, 29, 41]:
        test_theorem_B(p)


if __name__ == "__main__":
    demo()
