from qiskit import __version__, QuantumRegister,ClassicalRegister, QuantumCircuit, Aer, BasicAer, execute, transpile
from qiskit.tools.visualization import circuit_drawer, plot_histogram
import matplotlib.pyplot as plt
import numpy as np


backend_sim = BasicAer.get_backend('qasm_simulator')
backend_stat = BasicAer.get_backend('statevector_simulator')

M_sim = Aer.backends(name='qasm_simulator')[0]
S_sim = Aer.backends(name='statevector_simulator')[0]

q = QuantumRegister(1)
c=ClassicalRegister(1)

single_q_experiment = QuantumCircuit(q,c)
single_q_experiment.id(q[0])

single_q_experiment.measure(q,c)

job_M = execute(single_q_experiment,M_sim)
result = job_M.result()

counts = result.get_counts(single_q_experiment)
print("Final counts", counts)

print(circuit_drawer(single_q_experiment))

ident = ['|000>','|100>','|010>','|110>','|001>','|101>','|011>','|111>']

def wavefunction(qc,sim):
    job_S = execute(qc,sim)
    job_arr = np.array(job_S.result().get_statevector())
    for i in range(len(job_arr)):
        print(ident[i]+" "+str(job_arr[i]))

q = QuantumRegister(3)
c=ClassicalRegister(3)

qc = QuantumCircuit(q,c)
qc.h(q[0])
qc.h(q[1])
qc.h(q[2])

wavefunction(qc,S_sim)

# coś nowego 

def initialize_s(qc, qubits):
    for q in qubits:
        qc.h(q)
    return qc

n = 2

oracle = QuantumCircuit(n)
oracle = initialize_s(oracle, [0,1])
print(circuit_drawer(oracle))
wavefunction(oracle,S_sim)
oracle.cz(0,1)
print(circuit_drawer(oracle))
wavefunction(oracle,S_sim)

grover_circuit = QuantumCircuit(n)
grover_circuit = initialize_s(grover_circuit, [0,1])
grover_circuit.cz(0,1)
grover_circuit.h([0,1])
grover_circuit.x([0,1])
# grover_circuit.z([0,1])
grover_circuit.cz(0,1)
grover_circuit.x([0,1])
grover_circuit.h([0,1])
print(circuit_drawer(grover_circuit))

wavefunction(grover_circuit,S_sim)

qc = QuantumCircuit(3)
qc.h([2])
qc.ccx(0, 1,2)
qc.h([2])
R = qc.to_gate()
R.name = "R"

print(circuit_drawer(qc))

qc = QuantumCircuit(3)
qc.h([0,1,2])
qc.append(R,[0,1,2])
qc.h([0,1,2])
qc.x([0,1,2])
qc.append(R,[0,1,2])
qc.x([0,1,2])
qc.h([0,1,2])
print(circuit_drawer(qc))
qc.barrier()

qc.measure_all()
M_sim = Aer.backends(name='aer_simulator')[0]
job_sim = M_sim.run(transpile(qc,S_sim),shots=1024)
result = job_sim.result()
counts = result.get_counts(qc)
print(counts)
plt.figure()
plot_histogram(counts)
plt.show()

qc = QuantumCircuit(3)
qc.h([0,1,2])
qc.append(R,[0,1,2])
qc.h([0,1,2])
qc.x([0,1,2])
qc.append(R,[0,1,2])
qc.x([0,1,2])
qc.h([0,1,2])
print(circuit_drawer(qc))
qc.barrier()

qc.append(R,[0,1,2])
qc.h([0,1,2])
qc.x([0,1,2])
qc.append(R,[0,1,2])
qc.x([0,1,2])
qc.h([0,1,2])
print(circuit_drawer(qc))

qc.measure_all()
M_sim = Aer.backends(name='aer_simulator')[0]
job_sim = M_sim.run(transpile(qc,S_sim),shots=1024)
result = job_sim.result()
counts = result.get_counts(qc)
print(counts)
plt.figure()
plot_histogram(counts)
plt.show()


wavefunction(qc,S_sim)
