import numpy as np


def sieve_eratosthenes(n):
    is_prime = np.ones(n + 1, dtype=np.bool_)
    is_prime[:2] = False
    if n >= 4:
        is_prime[4::2] = False

    limit = int(np.sqrt(n)) + 1
    for p in range(3, limit, 2):
        if is_prime[p]:
            is_prime[p * p:n + 1:2 * p] = False

    return is_prime
