import json
import numpy as np

from braket.devices import LocalSimulator

from braket.ahs.atom_arrangement import AtomArrangement
from braket.ahs.hamiltonian import Hamiltonian
from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from braket.timings.time_series import TimeSeries
from braket.ahs.field import Field
from braket.ahs.pattern import Pattern
from braket.ahs.local_detuning import LocalDetuning
from braket.ahs.driving_field import DrivingField

from collections import Counter

megahertz = 1e+6
micron = 1e-6
microsecond = 1e-6
NUM_ATOMS = 4
# function hatFunc(t, k, clocks)

#     n = length(clocks)
#     if k > 1 && clocks[k-1] <= t <= clocks[k]
#         return (t - clocks[k-1]) / (clocks[k] - clocks[k-1])
#     elseif k < n && clocks[k] <= t <= clocks[k+1]
#         return (clocks[k+1]-t) / (clocks[k+1]-clocks[k])
#     else
#         return 0.0
#     end
# end

def hatFunc(t, k, clocks):
    n = len(clocks)

    if k > 0 and clocks[k-1] <= t <= clocks[k]:
        return (t - clocks[k-1]) / (clocks[k] - clocks[k-1])
    elif k < n and clocks[k] <= t <= clocks[k+1]:
        return (clocks[k+1]-t) / (clocks[k+1]-clocks[k])
    else:
        return 0.0


def pulseSchedule(t, params, clocks, initVal):

    numIntervals = int(len(params) / 2)
    levels = []

    for i in range(numIntervals):

        for j in [0,1]:

            levels.append(params[i]*initVal + params[numIntervals+i])

    vals = [0.0, *levels, 0.0]

    return np.sum([vals[k]*hatFunc(t, k, clocks) for k in range(len(vals))])


def pulseTimeSeries(params, clocks, initVal):
    numIntervals = int(len(params) / 2)
    timeSeries = TimeSeries()

    levels = []

    for i in range(numIntervals):

        for _ in [0,1]:
            levels.append(params[i]*initVal + params[numIntervals+i]*1e+6) # included conversion to megahertz on 2nd param

    vals = [0.0, *levels, 0.0]

    for c, v in zip(clocks, vals):
        timeSeries.put(c, v)

    return timeSeries

def simulatedResidualInference(dataPoint, params, latticeShape, latticeSpacing):

    latticeSpacing = latticeSpacing * micron # convert to microns
    minDt = 0.05 * microsecond # convert to microseconds
    numTiers = 3
    clocks = []

    # intake data point
    h_inputs = dataPoint[2:4]
    rabi_0 = dataPoint[1] * megahertz # convert to megahertz
    delta_global_0 = dataPoint[0] * megahertz
    delta_local_0 = dataPoint[4] * megahertz

    # intake parameters for analog calculation
    rabi_params = params[0:numTiers*2]
    global_detune_params = params[2*numTiers : 4*numTiers]
    local_detune_params = params[4*numTiers : 6*numTiers]
    h_params = params[6*numTiers:]


    for i in range(numTiers+1):
        for j in [0,1]:
            clocks.append( (4*i + j)*minDt )

    atom_register = AtomArrangement()

    if latticeShape == "square":
        # [(0.0, 0.0), (12.0, 0.0), (0.0, 12.0), (12.0, 12.0)]
        atom_register.add([0, 0])
        atom_register.add([latticeSpacing, 0])
        atom_register.add([0, -latticeSpacing])
        atom_register.add([latticeSpacing, -latticeSpacing])

    elif latticeShape == "triangle":
        # [(0.0, 0.0), (12.0, 0.0), (6.0, 10.392304845413264), (18.0, 10.392304845413264)]
        atom_register.add([0, 0])
        atom_register.add([latticeSpacing, 0])
        atom_register.add([latticeSpacing/2.0, -np.sqrt(latticeSpacing**2 - (latticeSpacing/2.0)**2) ])
        atom_register.add([latticeSpacing + latticeSpacing/2.0, -np.sqrt(latticeSpacing**2 - (latticeSpacing/2.0)**2)])

    else:
        raise Exception("Invalid Lattice shape.")

    H = Hamiltonian()

    ahs_program = AnalogHamiltonianSimulation(
        hamiltonian=H,
        register=atom_register
    )

    rabi_global_pulse = pulseTimeSeries(rabi_params, clocks, rabi_0)
    delta_global_pulse = pulseTimeSeries(global_detune_params, clocks, delta_global_0)

    
    
    drive = DrivingField(
        amplitude=rabi_global_pulse,
        phase=TimeSeries().from_lists(times=[0.0, clocks[-1]], values=[0.0, 0.0]),
        detuning=delta_global_pulse
    )

    H += drive

    # h_max = max(1.0, np.max(h_params))

    # h_params = h_params / h_max
    # h_inputs = h_inputs / h_max
    # local_detune_params = local_detune_params * h_max

    h_inputs[0] = max(h_inputs[0], 0)
    h_inputs[0] = min(h_inputs[0], 1)

    h_inputs[1] = max(h_inputs[1], 0)
    h_inputs[1] = min(h_inputs[1], 1)
    h_params[0] = min(h_params[0], 1)
    h_params[1] = min(h_params[1], 1)
    
    h_pattern = Pattern([h_inputs[0], h_params[0], h_inputs[1], h_params[1]])
    local_detune_pulse = pulseTimeSeries(local_detune_params, clocks, delta_local_0)

    local_detuning = LocalDetuning(
        magnitude=Field(
            time_series=local_detune_pulse,
            pattern=h_pattern
        )
    )

    H += local_detuning

    # show_drive_and_local_detuning(drive, local_detuning)
    # show_register(atom_register)
    
    device = LocalSimulator("braket_ahs")
   
    result_simulator = device.run(
       ahs_program,
       shots=10000,
        steps = 10000
    ).result() 

    # print(result_simulator)
    # print(ahs_program.to_ir().dict())
    label = -10
    logit = -10

    return result_simulator

def get_counts(result):
    """Aggregate state counts from AHS shot results

    A count of strings (of length = # of spins) are returned, where
    each character denotes the state of a spin (site):
        e: empty site
        u: up state spin
        d: down state spin

    Args:
        result (braket.tasks.analog_hamiltonian_simulation_quantum_task_result.AnalogHamiltonianSimulationQuantumTaskResult)

    Returns
        dict: number of times each state configuration is measured

    """
    state_counts = Counter()
    states = ['e', 'u', 'd']
    for shot in result.measurements:
        pre = shot.pre_sequence
        post = shot.post_sequence
        state_idx = np.array(pre) * (1 + np.array(post))
        state = "".join(map(lambda s_idx: states[s_idx], state_idx))
        state_counts.update((state,))
    return dict(state_counts)



def residualInferenceProgram(dataPoint, params, latticeShape, latticeSpacing):

    latticeSpacing = latticeSpacing * micron # convert to microns
    minDt = 0.05 * microsecond # convert to microseconds
    numTiers = 3
    clocks = []

    # intake data point
    h_inputs = dataPoint[2:4]
    rabi_0 = dataPoint[1] * megahertz # convert to megahertz
    delta_global_0 = dataPoint[0] * megahertz
    delta_local_0 = dataPoint[4] * megahertz

    # intake parameters for analog calculation
    rabi_params = params[0:numTiers*2]
    global_detune_params = params[2*numTiers : 4*numTiers]
    local_detune_params = params[4*numTiers : 6*numTiers]
    h_params = params[6*numTiers:]


    for i in range(numTiers+1):
        for j in [0,1]:
            clocks.append( (4*i + j)*minDt )

    atom_register = AtomArrangement()

    if latticeShape == "square":
        # [(0.0, 0.0), (12.0, 0.0), (0.0, 12.0), (12.0, 12.0)]
        atom_register.add([0, 0])
        atom_register.add([latticeSpacing, 0])
        atom_register.add([0, -latticeSpacing])
        atom_register.add([latticeSpacing, -latticeSpacing])

    elif latticeShape == "triangle":
        # [(0.0, 0.0), (12.0, 0.0), (6.0, 10.392304845413264), (18.0, 10.392304845413264)]
        atom_register.add([0, 0])
        atom_register.add([latticeSpacing, 0])
        atom_register.add([latticeSpacing/2.0, -np.sqrt(latticeSpacing**2 - (latticeSpacing/2.0)**2) ])
        atom_register.add([latticeSpacing + latticeSpacing/2.0, -np.sqrt(latticeSpacing**2 - (latticeSpacing/2.0)**2)])

    else:
        raise Exception("Invalid Lattice shape.")

    H = Hamiltonian()

    ahs_program = AnalogHamiltonianSimulation(
        hamiltonian=H,
        register=atom_register
    )

    rabi_global_pulse = pulseTimeSeries(rabi_params, clocks, rabi_0)
    delta_global_pulse = pulseTimeSeries(global_detune_params, clocks, delta_global_0)

    
    
    drive = DrivingField(
        amplitude=rabi_global_pulse,
        phase=TimeSeries().from_lists(times=[0.0, clocks[-1]], values=[0.0, 0.0]),
        detuning=delta_global_pulse
    )

    H += drive

    # h_max = max(1.0, np.max(h_params))

    # h_params = h_params / h_max
    # h_inputs = h_inputs / h_max
    # local_detune_params = local_detune_params * h_max
    
    h_pattern = Pattern([h_inputs[0], h_params[0], h_inputs[1], h_params[1]])
    local_detune_pulse = pulseTimeSeries(local_detune_params, clocks, delta_local_0)

    local_detuning = LocalDetuning(
        magnitude=Field(
            time_series=local_detune_pulse,
            pattern=h_pattern
        )
    )

    H += local_detuning



    return ahs_program
    