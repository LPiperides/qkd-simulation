import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Constants
BASES = ["+", "x"]  # Basis choices for SARG04
STATES = {
    "+": {0: "|0>", 1: "|1>"},
    "x": {0: "|+>", 1: "|->"},
}
BIT_VALUES = {
    v: k for basis in STATES.values() for k, v in basis.items()
}  # Reverse lookup for bit values


def prepare_qubit(bit: int, basis: str) -> QuantumCircuit:
    qc = QuantumCircuit(1, 1)
    if bit == 1:
        qc.x(0)
    if basis == "x":
        qc.h(0)
    return qc


def measure_qubit(qc, simulator, basis):
    if basis == "x":
        qc.h(0)
    qc.measure(0, 0)
    compiled_qc = transpile(qc, simulator)
    result = simulator.run(compiled_qc, shots=1).result()
    return int(max(result.get_counts(), key=result.get_counts().get))


def sarg04_simulation(num_bits: int, with_eve: bool = False) -> pd.DataFrame:
    simulator = AerSimulator()

    alice_bases = np.random.choice(BASES, num_bits)
    alice_bits = np.random.choice([0, 1], num_bits)
    alice_states = [STATES[alice_bases[i]][alice_bits[i]] for i in range(num_bits)]

    bob_bases = np.random.choice(BASES, num_bits)

    eve_bases = np.random.choice(BASES, num_bits) if with_eve else None
    eve_bits = np.empty(num_bits, dtype=int) if with_eve else None
    eve_states = [None] * num_bits if with_eve else None

    announced_states = []
    bob_bits = np.empty(num_bits, dtype=int)
    bob_states = []
    bob_discovered_states = []
    sifted_key = []

    eve_discovered_states = [] if with_eve else None
    eve_sifted_key = [] if with_eve else None

    for i in range(num_bits):
        qc = prepare_qubit(alice_bits[i], alice_bases[i])

        announced_computational = (
            STATES["+"][alice_bits[i]]
            if alice_bases[i] == "+"
            else STATES["+"][np.random.choice([0, 1])]
        )
        announced_hadamard = (
            STATES["x"][alice_bits[i]]
            if alice_bases[i] == "x"
            else STATES["x"][np.random.choice([0, 1])]
        )
        announced_states.append(f"{announced_computational} OR {announced_hadamard}")

        if with_eve:
            eve_bit = measure_qubit(qc.copy(), simulator, eve_bases[i])
            eve_bits[i] = eve_bit
            eve_state = STATES[eve_bases[i]][eve_bit]
            eve_states[i] = eve_state
            qc = prepare_qubit(eve_bit, eve_bases[i])

        bob_bit = measure_qubit(qc.copy(), simulator, bob_bases[i])
        bob_bits[i] = bob_bit
        bob_state = STATES[bob_bases[i]][bob_bit]
        bob_states.append(bob_state)

        if bob_state in (announced_computational, announced_hadamard):
            bob_discovered_states.append("")
            sifted_key.append("")
        else:
            discovered_state = (
                announced_hadamard if bob_bases[i] == "+" else announced_computational
            )
            bob_discovered_states.append(discovered_state)
            sifted_key.append(BIT_VALUES[discovered_state])

        if with_eve:
            if eve_state in (announced_computational, announced_hadamard):
                eve_discovered_states.append("")
                eve_sifted_key.append("")
            else:
                discovered_state = (
                    announced_hadamard
                    if eve_bases[i] == "+"
                    else announced_computational
                )
                eve_discovered_states.append(discovered_state)
                eve_sifted_key.append(BIT_VALUES[discovered_state])

    df_data = {
        "Bit_Number": np.arange(1, num_bits + 1),
        "Alice_State": alice_states,
        "Alice_Basis": alice_bases,
        "Alice_Bit": alice_bits,
        "Announced_States": announced_states,
    }

    if with_eve:
        df_data.update(
            {
                "Eve_State": eve_states,
                "Eve_Basis": eve_bases,
                "Eve_Bit": eve_bits,
                "Eve_Discovered_State": eve_discovered_states,
                "Eve_Sifted_Key": eve_sifted_key,
            }
        )

    df_data.update(
        {
            "Bob_State": bob_states,
            "Bob_Basis": bob_bases,
            "Bob_Bit": bob_bits,
            "Bob_Discovered_State": bob_discovered_states,
            "Sifted_Key": sifted_key,
        }
    )
    return pd.DataFrame(df_data)
