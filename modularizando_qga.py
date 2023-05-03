import os
import numpy as np
import quantum_mats as qm
from copy import deepcopy
from time import time

def initialize_population(init_population, n, cl, track_fidelity, store_path,generation_number):
    # Código que maneja la inicialización de la población y los errores relacionados
    if init_population is None:
        # generate a random population
        raise ValueError('init_population is required in this version. Random population is not supported.')
    elif type(init_population) == qm.rho:
        if not n or not cl:
            raise ValueError('If population is a quantum_matrix.rho object n and cl are required.')
        a = n//2
        ancillas = [ia for ia in range(n*cl, n*cl+a)]
        ancillas_initial_state = np.zeros((2 ** a, 2 ** a))
        ancillas_initial_state[0, 0] = 1
        rho_pop = qm.rho(np.kron(init_population.get_matrix(), ancillas_initial_state), dense=True)
    else:
        population = deepcopy(init_population)
        if n != len(population) or cl != len(population[0]):
            raise ValueError('n and cl mismatch. {}!={} or {}!={}.'.format(n, len(population), cl, len(population[0])))
        i = 0
        index = 0
        for p in population:
            for c in p:
                i += c * 2**(n*cl - index - 1)
                index += 1
        a = n//2
        ancillas = [ia for ia in range(n*cl, n*cl+a)]
        rho_pop = qm.rho([1], [i * 2**a], [i * 2**a], (2**(n*cl + a), 2**(n*cl + a)))

    if track_fidelity:
        fidelity_array = np.zeros(shape=(generation_number, 5, n, len(track_fidelity)))

    if store_path:
        store = open(store_path, 'w')
        print('Genetic Algorithm parameters:', file=store)
        print('generation_numbar = {:d}, pm = {:f}, population number = {:d}, chromosome length = {:d}'.format(
            generation_number, pm, n, cl), file=store)
        print('Tracked fidelity for states:', file=store)
        if track_fidelity:
            for state in track_fidelity:
                if type(state) == qm.rho:
                    print(state.get_matrix(), file=store)
                else:
                    print(repr(state), file=store)
        else:
            print("None", file=store)
        print(file=store)
    return rho_pop

def cloning_routine_unitary(n,cl,a):
    clone = qm.Identity(2 ** (n * cl))
    for nu in range(0, n // 2):
        s = qm.Swap_reg(range((nu + 1) * cl, (nu + 2) * cl),
                        range((nu + n // 2) * cl, (nu + 1 + n // 2) * cl),
                        n * cl)
        u = qm.Uclone(2 ** cl)
        unu = s.dot(qm.KronExpand(2 ** (nu * cl), u, 2 ** ((n - nu - 2) * cl))).dot(s)
        clone = clone.dot(unu)
    clone = qm.KronExpand(1, clone, 2 ** a)
    return clone.get_matrix()
def crossover_rutine_unitary(n,cl,a):
    # Build the crossover routine unitary
    cross_index = cl // 2
    qq1 = np.array([q1 for nu in range(0, n//2, 2) for q1 in range((n // 2 + nu) * cl + cross_index, (n // 2 + nu + 1) * cl)])
    qq2 = np.array([q2 for nu in range(0, n//2, 2) for q2 in range((n // 2 + nu + 1) * cl + cross_index, (n // 2 + nu + 2) * cl)])
    cross = qm.Swap_reg(qq1, qq2, n*cl)
    cross = qm.KronExpand(1, cross, 2 ** a)
    mat_cross = cross.get_matrix()

def build_unitary_routines(n, cl, fitness_basis, fitness_criteria):
    # Código para construir las rutinas unitarias
    # Build de cloning routine unitary
    mat_clone = cloning_routine_unitary(n,cl)

    # Build the crossover routine unitary
    cross_index = cl // 2
    qq1 = [q1 for nu in range(0, n//2, 2) for q1 in range((n // 2 + nu) * cl + cross_index, (n // 2 + nu + 1) * cl)]
    qq2 = [q2 for nu in range(0, n//2, 2) for q2 in range((n // 2 + nu + 1) * cl + cross_index, (n // 2 + nu + 2) * cl)]
    cross = qm.Swap_reg(qq1, qq2, n*cl)
    cross = qm.KronExpand(1, cross, 2 ** a)
    mat_cross = cross.get_matrix()

    # Build the sort/semi sort step unitary
    # You cant build it completely because you need to set the ancillas to
    # zero in each stage, but this makes it easier later.
    sort_arr = [np.identity(2 ** (n * cl + a)), np.identity(2 ** (n * cl + a))]
    for stage in [0, 1]:
        for reg in range(0, n - 1, 2):
            nu = reg + stage % 2
            if nu + 2 > n:
                continue
            reg1 = list(range(nu * cl, (nu + 1) * cl))
            reg2 = list(range((nu + 1) * cl, (nu + 2) * cl))
            ancilla = n * cl + reg // 2
            orac = qm.Oracle.get_qf_sort_oracle(nq=n * cl + a,
                                                reg1=reg1,
                                                reg2=reg2,
                                                ancilla=ancilla,
                                                uu=fitness_basis,
                                                criteria=fitness_criteria)

            cs = qm.CSwap_reg(ancilla,
                              reg1,
                              reg2,
                              n * cl + a).get_matrix()
            sort_arr[stage] = sort_arr[stage].dot(cs).dot(orac)
    sort0, sort1 = sort_arr
    mat_sort0 = sort0
    mat_sort1 = sort1
    return mat_clone, mat_cross, mat_sort0, mat_sort1


def build_mutation_arrays(mutation_unitary, pm, n, cl, mutation_pattern):
    # Código para construir los arreglos de mutación
    # Build mutation arrays
    if type(mutation_unitary) == list or type(pm) == list:
        if type(mutation_unitary) != list or type(pm) != list:
            raise ValueError("mutation_unitary and pm are not consistent.")
        if len(mutation_unitary) != len(pm):
            raise ValueError("The length of mutation_unitary and pm is not consistent.")
        if sum(pm) >= 1:
            raise ValueError("Mutation probabilities must add up to less than one.")
        if not (mutation_pattern is None):
            raise ValueError("Using mutation_pattern and a set of mutation unitaries is not compatible.")
        use_mutation_unitary_set = True
        pm_sum = sum(pm)
        pm_norm = [pmi / pm_sum for pmi in pm]
    else:
        pm_sum = pm
        use_mutation_unitary_set = False

    if not use_mutation_unitary_set:
        if mutation_unitary not in ["r", "R"]:
            if type(mutation_unitary) == str:
                if mutation_unitary in ["x", "X", "not", "NOT"]:
                    mutation_unitary = qm.rho([1, 1], [0, 1], [1, 0], (2, 2))
                elif mutation_unitary in ["h", "H"]:
                    mutation_unitary = qm.rho(np.array([1, 1, 1, -1])/np.sqrt(2), [0, 0, 1, 1], [0,  1, 0, 1], (2, 2))
                elif mutation_unitary in ["i", "I"]:
                    mutation_unitary = qm.Identity(2)
                else:
                    raise ValueError("mutation_unitary = \"%s\" not recognised." % mutation_unitary)
            else:
                mutation_unitary = qm.rho(mutation_unitary, dense=True)

            mut_arr = []
            mat_mut_arr = []
            for i in range(n * cl):
                mut_arr.append(qm.KronExpand(2 ** i, mutation_unitary, 2 ** (n * cl - i - 1 + a)))
                mat_mut_arr.append(np.kron(np.kron(np.identity(2 ** i),
                                                   mutation_unitary.get_matrix()),
                                           np.identity(2 ** (n * cl - i - 1 + a))))
    return use_mutation_unitary_set, pm_sum, pm_norm, mut_arr, mat_mut_arr


def build_pre_projection_rotation(pre_projection_unitary, n, cl):
    # Código para construir la rotación previa a la proyección
        # Build the pre-projection rotation
    if type(pre_projection_unitary) == str:
        if pre_projection_unitary in ["i", "I"]:
            pre_projection_unitary = "I"
        else:
            raise ValueError("pre_projection_unitary = \"%s\" not supported." % pre_projection_unitary)
    else:
        lower_rot_mat = np.identity(2 ** (n//2 * cl))
        for lreg in range(n-n//2):
            lower_rot_mat = np.kron(lower_rot_mat, pre_projection_unitary)
        lower_rot_mat = np.kron(lower_rot_mat, np.identity(2 ** a))

        lower_rot = None  # Not implemented
    return lower_rot_mat, lower_rot

def quantum_genetic_algorithm(init_population, n, cl, generation_number, pm, fitness_basis, fitness_criteria, mutation_unitary, pre_projection_unitary, store_path, track_fidelity):
    rho_pop, fidelity_array, store = initialize_population(init_population, n, cl, track_fidelity, store_path)
    mat_clone, mat_cross, mat_sort0, mat_sort1 = build_unitary_routines(n, cl, fitness_basis, fitness_criteria)
    use_mutation_unitary_set, pm_sum, pm_norm, mut_arr, mat_mut_arr = build_mutation_arrays(mutation_unitary, pm, n, cl, mutation_pattern)
    lower_rot_mat, lower_rot = build_pre_projection_rotation(pre_projection_unitary, n, cl)
    
    # Código para el bucle principal de generaciones y otras operaciones

def init_rho_pop_from_array(init_population, n, cl):
    population = np.array(deepcopy(init_population))
    if n != len(population) or cl != len(population[0]):
        raise ValueError('n and cl mismatch. {}!={} or {}!={}.'.format(n, len(population), cl, len(population[0])))

    i = 0
    index = 0
    for p in population:
        for c in p:
            i += c * 2 ** (n * cl - index - 1)
            index += 1

    a = n // 2
    ancillas = [ia for ia in range(n * cl, n * cl + a)]
    rho_pop = qm.rho([1], [i * 2 ** a], [i * 2 ** a], (2 ** (n * cl + a), 2 ** (n * cl + a)))
    return rho_pop,ancillas,a

def init_rho_pop_from_rho(init_population, n, cl):
    a = n // 2
    ancillas = [ia for ia in range(n * cl, n * cl + a)]
    ancillas_initial_state = np.zeros((2 ** a, 2 ** a))
    ancillas_initial_state[0, 0] = 1
    rho_pop = qm.rho(np.kron(init_population.get_matrix(), ancillas_initial_state), dense=True)
    return rho_pop,ancillas,a

def init_pop_numpy(init_population, n, cl, track_fidelity, store_path,generation_number):
    # Código que maneja la inicialización de la población y los errores relacionados
    if init_population is None:
        # generate a random population
        raise ValueError('init_population is required in this version. Random population is not supported.')
    elif type(init_population) == qm.rho:
        if not n or not cl:
            raise ValueError('If population is a quantum_matrix.rho object n and cl are required.')
        rho_pop,ancillas,a = init_rho_pop_from_rho(init_population, n, cl)
    else:
        rho_pop,ancillas,a = init_rho_pop_from_array(init_population, n, cl)
        

    if track_fidelity:
        fidelity_array = np.zeros(shape=(generation_number, 5, n, len(track_fidelity)))

    if store_path:
        store = open(store_path, 'w')
        print('Genetic Algorithm parameters:', file=store)
        print('generation_numbar = {:d}, pm = {:f}, population number = {:d}, chromosome length = {:d}'.format(
            generation_number, pm, n, cl), file=store)
        print('Tracked fidelity for states:', file=store)
        if track_fidelity:
            for state in track_fidelity:
                if type(state) == qm.rho:
                    print(state.get_matrix(), file=store)
                else:
                    print(repr(state), file=store)
        else:
            print("None", file=store)
        print(file=store)
    return rho_pop,ancillas
