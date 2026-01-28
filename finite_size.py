def finite_size_penalty(N):
    if N < 1e5:
        return 0.75
    elif N < 1e6:
        return 0.85
    elif N < 1e7:
        return 0.90
    else:
        return 0.95


def finite_size_skr(asymptotic_skr, N):
    penalty = finite_size_penalty(N)
    return asymptotic_skr * penalty
