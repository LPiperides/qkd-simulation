"""Quantum channel noise model.

Every hop a qubit makes (Alice to Bob, Alice to Eve and Eve to Bob, or from
E91's pair source to either party) is modelled as a single-qubit depolarizing
channel

    E(rho) = (1 - p) rho + p I/2,

i.e. with probability ``p`` the transmitted qubit is replaced by the maximally
mixed state. For a qubit measured in its own preparation basis this flips the
bit with probability ``p/2``; measured in the conjugate basis the outcome is
uniformly random regardless of ``p``.

The error is attached to the ``id`` instruction that :func:`prepare_qubit`
appends to every circuit rather than to the preparation gates. That choice
matters: the state |0> is prepared with *no* gates, so a noise model attached
to ``x`` and ``h`` would leave a quarter of all transmissions noise-free and
skew every QBER estimate. With the identity hook each hop applies exactly one
depolarizing event to every qubit, and the analytic QBER(p) formulas in
``docs/theory.md`` hold exactly.
"""

from __future__ import annotations

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from .primitives import CHANNEL_GATE


def depolarizing_noise_model(p: float) -> NoiseModel | None:
    """Return an Aer noise model for a depolarizing channel of strength ``p``.

    Returns ``None`` for ``p == 0`` so callers can build a plain, faster
    noiseless simulator. Raises ``ValueError`` outside ``[0, 1]``.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"depolarizing probability must lie in [0, 1], got {p}")
    if p == 0.0:
        return None

    model = NoiseModel()
    model.add_all_qubit_quantum_error(depolarizing_error(p, 1), [CHANNEL_GATE])
    return model


def make_simulator(noise: float = 0.0, **options) -> AerSimulator:
    """Build an ``AerSimulator`` with a depolarizing channel of strength ``noise``.

    Extra keyword arguments are forwarded to ``AerSimulator``.
    """
    model = depolarizing_noise_model(noise)
    if model is None:
        return AerSimulator(**options)
    return AerSimulator(noise_model=model, **options)
