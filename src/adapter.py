import numpy as np


def adapter(h, accept_probs, target_accept_prob, method="none", lr=0.1, batch_size=30):
    # Algorithm 4: AM aglorithm with global adaptive scaling
    if method == "algo4":
        return stochastic_approximation_adapter(
            h, accept_probs[-1], target_accept_prob, lr
        )
    elif method == "batch":
        return batch_adapter(h, accept_probs, target_accept_prob, lr, batch_size)

    return h


def stochastic_approximation_adapter(h, last_accept_prob, target_accept_prob, lr=0.6):
    h *= np.exp(lr * (last_accept_prob - target_accept_prob))
    return h


def batch_adapter(h, accept_probs, target_accept_prob, lr=0.1, batch_size=30):
    if len(accept_probs) % batch_size == 0:
        diff = accept_probs[-1] - target_accept_prob

        if np.abs(diff) > 0.05:
            recent_accept_probs = accept_probs[-batch_size:]
            mean_accept_prob = sum(recent_accept_probs) / len(recent_accept_probs)

            h *= np.exp(
                lr
                * (diff)
                * (1 + np.sqrt(np.abs(mean_accept_prob - target_accept_prob)))
            )

    return h
