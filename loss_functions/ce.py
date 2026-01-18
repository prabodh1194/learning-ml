"""
Multiclass cross entropy loss
L = -(1/n) * Σ[y_c * log(y_c_pred)]
"""

import numpy as np
import mlx.core as mx
import torch

from loss_functions.base import LossFunction, test_loss


class CE(LossFunction):
    class np:
        @staticmethod
        def forward(predictions: np.ndarray, targets: np.ndarray) -> float:
            logits = predictions
            logits_maxes = logits.max(axis=1, keepdims=True)
            logits_stable = logits - logits_maxes
            logits_exp = np.exp(logits_stable)
            logits_exp_norm = logits_exp.sum(axis=1, keepdims=True)

            p = logits_exp / logits_exp_norm

            n_samples = targets.shape[0]
            loss = -(targets * np.log(p)).sum() / n_samples

            return loss

        @staticmethod
        def backward(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
            """
            Manual backprop through: logits → softmax → cross-entropy loss

            Forward pass recap:
                z_stable = z - max(z)           # numerical stability
                exp_z = exp(z_stable)           # exponentiate
                sum_exp = sum(exp_z)            # denominator
                p = exp_z / sum_exp             # softmax probabilities
                L = -1/n · Σ t_i · log(p_i)     # cross-entropy loss

            We'll backprop through each step to get dL/dz.
            Spoiler: it all collapses to (p - t) / n
            """
            n_samples = targets.shape[0]
            logits = predictions

            # ==================== FORWARD (recompute for backprop) ====================

            z_stable = logits - logits.max(axis=1, keepdims=True)
            exp_z = np.exp(z_stable)
            sum_exp = exp_z.sum(axis=1, keepdims=True)
            p = exp_z / sum_exp

            # ==================== BACKWARD ====================

            # --- Step 1: dL/dp ---
            # L = -1/n · Σ t_i · log(p_i)
            # ∂L/∂p_i = -t_i / (n · p_i)

            dL_dp = -targets / (n_samples * p)

            # --- Step 2: dp/d(exp_z) ---
            #
            # Setup: we have 3 classes, so:
            #   exp_z = [exp_z_0, exp_z_1, exp_z_2]
            #   sum_exp = exp_z_0 + exp_z_1 + exp_z_2
            #   p_0 = exp_z_0 / sum_exp
            #   p_1 = exp_z_1 / sum_exp
            #   p_2 = exp_z_2 / sum_exp
            #
            # Key question: how does changing exp_z_k affect each probability p_i?
            #
            # Example: how does exp_z_1 affect things?
            #
            #   p_0 = exp_z_0 / (exp_z_0 + exp_z_1 + exp_z_2)
            #         ↑ numerator doesn't contain exp_z_1
            #         ↑ but denominator does!
            #         If exp_z_1 goes up, denominator grows, so p_0 goes DOWN.
            #
            #   p_1 = exp_z_1 / (exp_z_0 + exp_z_1 + exp_z_2)
            #         ↑ numerator contains exp_z_1
            #         ↑ denominator also contains exp_z_1
            #         If exp_z_1 goes up, both grow, but numerator wins, so p_1 goes UP.
            #
            #   p_2 = exp_z_2 / (exp_z_0 + exp_z_1 + exp_z_2)
            #         ↑ same as p_0: numerator doesn't have exp_z_1, denominator does.
            #         If exp_z_1 goes up, p_2 goes DOWN.
            #
            # So increasing exp_z_1 increases p_1, but decreases p_0 and p_2.
            # This makes sense: probabilities must sum to 1.
            # If one goes up, others must go down to compensate.
            #
            # Now let's get the actual derivatives.
            #
            # Case A: i = k (how does exp_z_i affect its own probability p_i?)
            #
            #   p_i = exp_z_i / sum_exp
            #
            #   Using quotient rule: d/dx [f/g] = (f'g - fg') / g²
            #     f = exp_z_i      → df/d(exp_z_i) = 1
            #     g = sum_exp      → dg/d(exp_z_i) = 1  (because sum_exp contains exp_z_i)
            #
            #   ∂p_i/∂exp_z_i = (1 · sum_exp - exp_z_i · 1) / sum_exp²
            #                 = (sum_exp - exp_z_i) / sum_exp²
            #                 = 1/sum_exp - exp_z_i/sum_exp²
            #                 = 1/sum_exp - p_i/sum_exp
            #                 = (1 - p_i) / sum_exp
            #
            # Case B: i ≠ k (how does exp_z_k affect a different probability p_i?)
            #
            #   p_i = exp_z_i / sum_exp
            #
            #   Using quotient rule again:
            #     f = exp_z_i      → df/d(exp_z_k) = 0  (exp_z_i doesn't contain exp_z_k)
            #     g = sum_exp      → dg/d(exp_z_k) = 1  (sum_exp does contain exp_z_k)
            #
            #   ∂p_i/∂exp_z_k = (0 · sum_exp - exp_z_i · 1) / sum_exp²
            #                 = -exp_z_i / sum_exp²
            #                 = -p_i / sum_exp
            #
            # Summary:
            #   ∂p_i/∂exp_z_k = (1 - p_i) / sum_exp    when i = k
            #   ∂p_i/∂exp_z_k = -p_i / sum_exp         when i ≠ k
            #
            # Now we apply chain rule to get dL/d(exp_z_k):
            #
            #   dL/d(exp_z_k) = sum over all i of: (dL/dp_i) · (∂p_i/∂exp_z_k)
            #
            # Let's expand this for k=1 with 3 classes:
            #
            #   dL/d(exp_z_1) = (dL/dp_0) · (∂p_0/∂exp_z_1)      # i=0, k=1, different
            #                 + (dL/dp_1) · (∂p_1/∂exp_z_1)      # i=1, k=1, same
            #                 + (dL/dp_2) · (∂p_2/∂exp_z_1)      # i=2, k=1, different
            #
            #                 = (dL/dp_0) · (-p_0/sum_exp)
            #                 + (dL/dp_1) · ((1-p_1)/sum_exp)
            #                 + (dL/dp_2) · (-p_2/sum_exp)
            #
            # We can split this into two parts:
            #
            #   Part 1 (the i=k term): (dL/dp_1) · (1/sum_exp)
            #   Part 2 (all terms):    -1/sum_exp · [dL/dp_0 · p_0 + dL/dp_1 · p_1 + dL/dp_2 · p_2]
            #
            # Rewriting more generally for any k:
            #
            #   dL/d(exp_z_k) = (dL/dp_k) / sum_exp                    # direct effect
            #                 - (1/sum_exp) · Σ_i (dL/dp_i · p_i)      # indirect effect
            #
            # The direct effect: exp_z_k directly increases p_k (numerator effect)
            # The indirect effect: exp_z_k increases sum_exp, which decreases ALL probabilities
            #                      (denominator effect). We weight by how much we care about
            #                      each probability (dL/dp_i) times how big it is (p_i).

            # Direct effect: each exp_z_k pushes up its own p_k
            dL_dexp_direct = dL_dp / sum_exp

            # Indirect effect: each exp_z_k also makes the denominator bigger,
            # which pushes down ALL probabilities (including p_k itself).
            # The amount we care about this is weighted by dL/dp_i · p_i for each i.
            dL_dexp_indirect = -(dL_dp * p).sum(axis=1, keepdims=True) / sum_exp

            dL_dexp = dL_dexp_direct + dL_dexp_indirect

            # --- Step 3: d(exp_z)/d(z_stable) ---
            # exp_z = exp(z_stable)
            # derivative of exp is exp

            dL_dz_stable = exp_z * dL_dexp

            # --- Step 4: d(z_stable)/d(logits) ---
            # z_stable = z - max(z)
            #
            # Technically max(z) depends on z, so there's a subtle gradient here.
            # But in practice, shifting by a constant doesn't change softmax output,
            # and the gradient w.r.t. the max term cancels out.
            # So: dL/dz = dL/dz_stable

            dL_dz = dL_dz_stable

            # All that work should equal: (p - t) / n
            # The "funky" derivation magically simplifies because softmax + CE
            # are natural partners (exponential family likelihood).

            return dL_dz

    class mlx:
        @staticmethod
        def forward(predictions: mx.array, targets: mx.array) -> float:
            np.exp = mx.exp
            np.log = mx.log
            return CE.np.forward(predictions, targets)

        @staticmethod
        def backward(predictions: mx.array, targets: mx.array) -> mx.array:
            np.exp = mx.exp
            return CE.np.backward(predictions, targets)

    class torch:
        @staticmethod
        def forward(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.cross_entropy(predictions, targets)


if __name__ == "__main__":
    test_loss(
        CE,
        predictions=np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3], [0.2, 0.8, 1.5]]),
        targets=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    )
