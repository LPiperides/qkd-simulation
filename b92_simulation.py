# Imports
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Constants
BASES = ["+", "x"]  # Basis choices for B92
STATES = {
    "+": {0: "|0>", 1: "|1>"},
    "x": {0: "|+>", 1: "|->"},
}


def prepare_qubit(bit: int, basis: str) -> QuantumCircuit:
    """
    Prepares a qubit in the specified basis.

    Parameters:
        bit (int): The bit value (0 or 1) to encode.
        basis (str): The encoding basis ('+' for computational, 'x' for diagonal).

    Returns:
        QuantumCircuit: The prepared qubit circuit.
    """
    qc = QuantumCircuit(1, 1)

    if bit == 1:
        qc.x(0)  # Apply X gate for bit 1

    if basis == "x":
        qc.h(0)  # Apply Hadamard gate for diagonal basis

    return qc


def measure_qubit(qc: QuantumCircuit, simulator: AerSimulator, basis: str) -> int:
    """
    Measures a qubit in the specified basis using Qiskit's AerSimulator.

    Parameters:
        qc (QuantumCircuit): The quantum circuit containing the qubit to be measured.
        simulator (AerSimulator): The Qiskit Aer simulator instance.
        basis (str): The measurement basis ('+' for computational, 'x' for diagonal).

    Returns:
        int: The measured bit (0 or 1).
    """
    if basis == "x":
        qc.h(0)  # Apply Hadamard before measurement in diagonal basis

    qc.measure(0, 0)

    # Transpile and execute the circuit
    compiled_qc = transpile(qc, simulator)
    job = simulator.run(compiled_qc, shots=1)
    result = job.result()
    counts = result.get_counts()

    # Extract the most frequent measurement outcome
    return int(max(counts, key=counts.get, default="0"))


def b92_simulation(num_bits: int, with_eve: bool = False) -> pd.DataFrame:
    """
    Simulates the B92 quantum key distribution protocol.

    Parameters:
        num_bits (int): Number of bits to generate.
        with_eve (bool): Whether to include an eavesdropper (Eve).

    Returns:
        pd.DataFrame: Contains bit number, bases, sent bits, measured bits (and Eve's bits if with_eve=True).
    """
    # Initialize Alice's bits and basis choices
    alice_bits = np.random.choice([0, 1], num_bits)
    alice_bases = np.where(alice_bits == 0, "+", "x")  # '+' for 0, 'x' for 1
    alice_states = [STATES[alice_bases[i]][alice_bits[i]] for i in range(num_bits)]

    # Bob randomly chooses bases
    bob_bases = np.random.choice(BASES, num_bits)
    bob_bits, bob_states = [], []

    # Initialize Eve's values if needed
    eve_bases = np.random.choice(BASES, num_bits) if with_eve else None
    eve_bits, eve_states = ([] if with_eve else None), ([] if with_eve else None)

    # Initialize the quantum simulator
    simulator = AerSimulator()

    for i in range(num_bits):
        # Alice prepares her qubit
        qc = prepare_qubit(alice_bits[i], alice_bases[i])

        # Eve intercepts, measures, and resends (if enabled)
        if with_eve:
            eve_bit = measure_qubit(qc, simulator, eve_bases[i])
            eve_bits.append(eve_bit)
            eve_state = STATES[eve_bases[i]][eve_bit]
            eve_states.append(eve_state)

            # Eve resends the qubit based on her measurement
            qc = prepare_qubit(eve_bit, eve_bases[i])

        # Bob measures the qubit in his chosen basis
        bob_bit = measure_qubit(qc, simulator, bob_bases[i])
        bob_bits.append(bob_bit)
        bob_state = STATES[bob_bases[i]][bob_bit]
        bob_states.append(bob_state)

    # Construct the results DataFrame
    df_data = {
        "Bit_Number": np.arange(1, num_bits + 1),
        "Alice_State": alice_states,
        "Alice_Basis": alice_bases,
        "Alice_Bit": alice_bits,
    }

    if with_eve:
        df_data.update(
            {
                "Eve_State": eve_states,
                "Eve_Basis": eve_bases,
                "Eve_Bit": eve_bits,
            }
        )

    df_data.update(
        {
            "Bob_State": bob_states,
            "Bob_Basis": bob_bases,
            "Bob_Bit": bob_bits,
        }
    )

    return pd.DataFrame(df_data)
