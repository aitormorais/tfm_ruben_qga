import os
import numpy as np
import quantum_mats as qm
from copy import deepcopy
import time
import re
import cProfile
import pstats
from functools import lru_cache

def eliminar_contenido(archivo):
    """Funcion que se asegura de que el archivo.txt donde escribiremos los tiempos este limpio"""
    if os.path.exists(archivo):
        print(f"El archivo {archivo} existe. Eliminando su contenido.")
        open(archivo, "w").close()
    else:
        print(f"El archivo {archivo} no existe.")

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
    return cross.get_matrix()

def sor_semisort_unitary_og(n,cl,a, fitness_basis, fitness_criteria):
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
    return mat_sort0,mat_sort1

def sort_semisort_unitary(n,cl,a, fitness_basis, fitness_criteria):
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
    return sort0,sort1

def validate_mutation_unitary(mutation_unitary):
    if re.match(r"^[rR]$", mutation_unitary):
        return True
    elif re.match(r"^(?:[xX]|not|NOT)$", mutation_unitary):
        return True
    elif re.match(r"^[hH]$", mutation_unitary):
        return True
    elif re.match(r"^[iI]$", mutation_unitary):
        return True
    return False

def create_mutation_unitary(mutation_unitary):
    if re.match(r"^[rR]$", mutation_unitary):
        return qm.rho(mutation_unitary, dense=True)
    elif re.match(r"^(?:[xX]|not|NOT)$", mutation_unitary):
        return qm.rho([1, 1], [0, 1], [1, 0], (2, 2))
    elif re.match(r"^[hH]$", mutation_unitary):
        return qm.rho(np.array([1, 1, 1, -1])/np.sqrt(2), [0, 0, 1, 1], [0,  1, 0, 1], (2, 2))
    elif re.match(r"^[iI]$", mutation_unitary):
        return qm.Identity(2)
    raise ValueError("mutation_unitary = \"%s\" not recognised." % mutation_unitary)

def build_unitary_routines(n, cl, fitness_basis, fitness_criteria,a):
    # Código para construir las rutinas unitarias
    # Build de cloning routine unitary
    mat_clone = cloning_routine_unitary(n,cl,a)
    # Build the crossover routine unitary
    mat_cross = crossover_rutine_unitary(n,cl,a)

    # Build the sort/semi sort step unitary
    # You cant build it completely because you need to set the ancillas to
    # zero in each stage, but this makes it easier later.
    mat_sort0,mat_sort1 = sort_semisort_unitary(n,cl,a, fitness_basis, fitness_criteria)

    return mat_clone, mat_cross, mat_sort0, mat_sort1

def build_mutation_arrays(mutation_unitary, pm, n, cl, mutation_pattern):
    # Código para construir los arreglos de mutación
    # Build mutation arrays
    mut_arr = []
    mat_mut_arr = []
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

            #mut_arr = []
            #mat_mut_arr = []
            for i in range(n * cl):
                mut_arr.append(qm.KronExpand(2 ** i, mutation_unitary, 2 ** (n * cl - i - 1 + a)))
                mat_mut_arr.append(np.kron(np.kron(np.identity(2 ** i),
                                                   mutation_unitary.get_matrix()),
                                           np.identity(2 ** (n * cl - i - 1 + a))))
    return use_mutation_unitary_set, pm_sum, pm_norm, mut_arr, mat_mut_arr

def build_pre_projection_rotation(pre_projection_unitary, n, cl,a):
    """
    Build the pre-projection rotation matrix.

    :param pre_projection_unitary: string or unitary matrix for pre-projection
    :param n: number of qubits
    :param cl: control lines
    :param a: ancillary qubits
    :return: pre-projection rotation matrix
    """
    # Código para construir la rotación previa a la proyección
        # Build the pre-projection rotation
    lower_rot_mat = np.identity(2 ** (n//2 * cl))
    for lreg in range(n-n//2):
        lower_rot_mat = np.kron(lower_rot_mat, pre_projection_unitary)
    lower_rot_mat = np.kron(lower_rot_mat, np.identity(2 ** a))

    lower_rot = None  # Not implemented
    return lower_rot_mat

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

def init_pop_numpy(init_population, n, cl,generation_number):
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
    return rho_pop,ancillas,a

def apply_mutation(rho_pop, n, cl, a, pm_sum, pm_norm, mutation_unitary,pm):
    """
    Aplica una mutación a la población con una probabilidad dada.
    
    :param rho_pop: Población actual
    :param n: Número de qubits
    :param cl: Número de cromosomas
    :param a: Cantidad a agregar al exponente de la matriz de identidad
    :param pm_sum: Probabilidad total de mutación
    :param pm_norm: Probabilidades normalizadas para cada mutación
    :param mutation_unitary: Matrices unitarias de mutación
    :return: Población mutada
    """
    for c in range(n * cl):
        r = np.random.random()
        if r < pm_sum:
            mu = mutation_unitary[np.random.choice(range(len(pm)), p=pm_norm)]
            mut_mat = np.kron(np.kron(np.identity(2 ** c), mu),
                              np.identity(2 ** (n * cl - c - 1 + a)))
            rho_pop = qm.rho(mut_mat.dot(rho_pop.get_matrix().dot(np.transpose(mut_mat).conjugate())),
                             dense=True)
    return rho_pop

def apply_projection(rho_pop, n, cl, a, projection_method, pre_projection_unitary):
    """
    Aplica la proyección deseada a la población.

    :param rho_pop: Población actual
    :param n: Número de qubits
    :param cl: Número de cromosomas
    :param a: Cantidad a agregar al exponente de la matriz de identidad
    :param projection_method: Método de proyección a utilizar
    :param pre_projection_unitary: Matriz unitaria a aplicar antes de la proyección
    :return: Población después de la proyección
    """
    if projection_method != 'ptrace':
        for q in range(n // 2 * cl, n * cl):
            rho_pop.projection_controlled_rotation(q, 1, qm.rho([1, 1], [0, 1], [1, 0], (2, 2)),
                                                   pre_projection_unitary, projection_method)
    else:
        if type(pre_projection_unitary) != str:
            rho_pop = qm.rho(lower_rot_mat.dot(rho_pop.get_matrix()).dot(np.transpose(np.conjugate(lower_rot_mat))), dense=True)
        elif pre_projection_unitary != "I":
            raise Exception("Only \"I\" pre_projection_unitary is supported with type str")

        rho_pop = rho_pop.partial_trace(list(range(n // 2 * cl)))
        rho_pop = qm.rho(np.kron(rho_pop.get_matrix(),
                                 qm.rho([1], [0], [0],
                                        (2 ** (n * cl - n // 2 * cl + a),
                                         2 ** (n * cl - n // 2 * cl + a))).get_matrix()), dense=True)
    return rho_pop

def apply_stage_transform(rho_pop, n, stage, sort0, sort1, mat_sort0, mat_sort1, ancillas, projection_method, pre_projection_unitary, a,cl):
    """
    Aplica la transformación de etapa a la población.

    :param rho_pop: Población actual
    :param n: Número de qubits
    :param stage: Etapa actual del proceso
    :param sort0: Matriz de ordenamiento 0
    :param sort1: Matriz de ordenamiento 1
    :param mat_sort0: Matriz de ordenamiento para stage % 2 == 0
    :param mat_sort1: Matriz de ordenamiento para stage % 2 == 1
    :param ancillas: Lista de ancillas
    :param projection_method: Método de proyección a utilizar
    :param pre_projection_unitary: Matriz unitaria a aplicar antes de la proyección
    :param a: Cantidad a agregar al exponente de la matriz de identidad
    :return: Población después de la transformación de etapa
    """
    sort = [sort0, sort1][stage % 2]
    mat_sort = [mat_sort0, mat_sort1][stage % 2]
    
    rho_pop = qm.rho(mat_sort.dot(rho_pop.get_matrix()).dot(np.transpose(mat_sort).conjugate()), dense=True)

    if projection_method != 'ptrace':
        for ai in ancillas:
            rho_pop.projection_controlled_rotation(ai, 1, qm.rho([1, 1], [0, 1], [1, 0], (2, 2)),
                                                pre_projection_unitary, projection_method)
    else:
        rho_pop = rho_pop.partial_trace(list(range(n * cl)))#intentar numpy con esto
        rho_pop = qm.rho(np.kron(rho_pop.get_matrix(),
                                qm.rho([1], [0], [0], (2 ** a, 2 ** a)).get_matrix()), dense=True)




    return rho_pop

def anotar_tiempos(inicio,fin,nombre_archivo,nombre_funcion):
    """Anotar en un archivo.txt los tiempos de cada parte
    :param inicio: momento en el que se inicio
    :param fin: momento en el que finaliza
    :param nombre_archivo: cadena de texto que representa el nombre del archivo donde se guarda la info
    :param nombre_funcion : cadena de texto con el nombre del metodo que medimos
    """
    ejecucion = fin - inicio
    archivo = open(nombre_archivo,"a")
    archivo.write(nombre_funcion+": "+str(ejecucion)+" \n ")
    archivo.close()

def quantum_genetic_algorithm(fitness_criteria, fitness_basis=None,init_population=None, n=None, cl=None,generation_number=100, pm=0.01, mutation_pattern=None, mutation_unitary="x",projection_method="r", pre_projection_unitary="I",store_path=None, track_fidelity=None, track_only_reg_states=True):
    inicio= time.time()
    rho_pop,ancillas,a =init_pop_numpy(init_population, n, cl, generation_number)
    fin = time.time()
    #1
    anotar_tiempos(inicio, fin, "modulos.txt", "init_pop")
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
    inicio = time.time()
    mat_clone, mat_cross, sort0, sort1=build_unitary_routines(n, cl, fitness_basis, fitness_criteria,a)
    fin = time.time()
    anotar_tiempos(inicio, fin,  "modulos.txt", "build_unitary_routines")
    inicio = time.time()
    use_mutation_unitary_set, pm_sum, pm_norm, mut_arr, mat_mut_arr = build_mutation_arrays(mutation_unitary, pm, n, cl, mutation_pattern)
    fin = time.time()
    anotar_tiempos(inicio, fin, "modulos.txt", "build_mutation_arrays")
    inicio = time.time()
    if isinstance(pre_projection_unitary, str):
        if re.match(r"^[iI]$", pre_projection_unitary):
            pre_projection_unitary = "I"
        else:
            raise ValueError(f'pre_projection_unitary = "{pre_projection_unitary}" not supported.')
    else:
        lower_rot_mat = build_pre_projection_rotation(pre_projection_unitary, n, cl,a)

    fin = time.time()
    anotar_tiempos(inicio, fin, "modulos.txt", "preprojection_rotation")
    for generation in range(generation_number):
        if track_fidelity:
            for reg in range(n):
                reg_state = rho_pop.partial_trace(list(range(reg * cl, (reg + 1) * cl)))
                for i, state in enumerate(track_fidelity):
                    fidelity_array[generation, 0, reg, i] = reg_state.fidelity(state)
        if store_path:
            print('-'*32, '\n',
                  ' '*9, 'Generation: {}'.format(generation),
                  '\n', '-' * 32, '\n',
                  file=store, sep='')

            if not track_only_reg_states:
                print('initial state, population + ancillas:\n', repr(rho_pop.store), file=store)
                print(file=store)
            else:
                print('initial state:', file=store)
                for reg in range(n):
                    print('register {:2d}'.format(reg), end=' '*4, file=store)
                    print(repr(rho_pop.partial_trace(list(range(reg * cl, (reg + 1) * cl))).get_matrix()), file=store)
                    print('\n', file=store)
        
        inicio = time.time()
        for stage in range(0, n):
            hasiera = time.time()
            rho_pop = apply_stage_transform(rho_pop, n, stage, sort0, sort1, sort0, sort1, ancillas, projection_method, pre_projection_unitary, a,cl)
            amaiera = time.time()
            anotar_tiempos(hasiera, amaiera, "modulos.txt", "apply_stage_transform_for_stage")
        
        fin = time.time()
        anotar_tiempos(inicio, fin, "modulos.txt", "preprojection_rotation_for_generation")
        if track_fidelity:
            for reg in range(n):
                reg_state = rho_pop.partial_trace(list(range(reg * cl, (reg + 1) * cl)))
                for i, state in enumerate(track_fidelity):
                    fidelity_array[generation, 1, reg, i] = reg_state.fidelity(state)
        if store_path:
            if not track_only_reg_states:
                print('sorted state, population + ancillas:\n', repr(rho_pop.store), file=store)
            else:
                print('sorted state:', file=store)
                for reg in range(n):
                    print('register {:2d}'.format(reg), end=' ' * 4, file=store)
                    print(repr(rho_pop.partial_trace(list(range(reg * cl, (reg + 1) * cl))).get_matrix()), file=store)
                    print('\n', file=store)
            print(file=store)
        inicio = time.time()   
        rho_pop = apply_projection(rho_pop, n, cl, a, projection_method, pre_projection_unitary)
        fin = time.time()
        anotar_tiempos(inicio, fin, "modulos.txt", "apply_projection")
        if track_fidelity:
            for reg in range(n):
                reg_state = rho_pop.partial_trace(list(range(reg * cl, (reg + 1) * cl)))
                for i, state in enumerate(track_fidelity):
                    fidelity_array[generation, 2, reg, i] = reg_state.fidelity(state)
        if store_path:
            if not track_only_reg_states:
                print('state after sort and clear, population + ancillas:\n', repr(rho_pop.store), file=store)
            else:
                print('state after sort and clear:', file=store)
                for reg in range(n):
                    print('register {:2d}'.format(reg), end=' ' * 4, file=store)
                    print(repr(rho_pop.partial_trace(list(range(reg * cl, (reg + 1) * cl))).get_matrix()), file=store)
                    print('\n', file=store)
            print(file=store)

            # 4 - Crossover to fill
        inicio = time.time()
        rho_pop = qm.rho(mat_clone.dot(rho_pop.get_matrix()).dot(np.transpose(mat_clone)), dense=True)
        rho_pop = qm.rho(mat_cross.dot(rho_pop.get_matrix()).dot(np.transpose(mat_cross)), dense=True)
        fin = time.time()
        anotar_tiempos(inicio, fin, "modulos.txt", "crossover_to_fill")
        if track_fidelity:
            for reg in range(n):
                reg_state = rho_pop.partial_trace(list(range(reg * cl, (reg + 1) * cl)))
                for i, state in enumerate(track_fidelity):
                    fidelity_array[generation, 3, reg, i] = reg_state.fidelity(state)
        if store_path:
            if not track_only_reg_states:
                print('state after crossover, population + ancillas:\n', repr(rho_pop.store), file=store)
            else:
                print('state after crossover:', file=store)
                for reg in range(n):
                    print('register {:2d}'.format(reg), end=' ' * 4, file=store)
                    print(repr(rho_pop.partial_trace(list(range(reg * cl, (reg + 1) * cl))).get_matrix()), file=store)
                    print('\n', file=store)
            print(file=store)
        inicio = time.time()
        rho_pop = apply_mutation(rho_pop, n, cl, a, pm_sum, pm_norm, mutation_unitary,pm)
        fin = time.time()
        anotar_tiempos(inicio, fin, "modulos.txt", "mutation")

    if store_path:
        store.close()
    if track_fidelity:
        return rho_pop, fidelity_array
    else:
        return rho_pop

#main
def qga_qf_test(fitness_states, samples, dirpath):
    """
    Not updated to QGA_QF.
    :return:
    """
    g = 10
    n = 4
    cl = 2
    pm = [1 / n / cl / 3] * 3
    ppu = "I"
    mu = [np.array([[0, 1], [1, 0]]),
          np.array([[0, -1j], [1j, 0]]),
          np.array([[1, 0], [0, -1]])]
    new_dirpath = dirpath

    i = 1
    while os.path.exists(new_dirpath) and i < 1000:
        if dirpath.split("_")[-1].isnumeric():
            if i == 1:
                i = int(dirpath.split("_")[-1]) + 1
                new_dirpath = "_".join(dirpath.split("_")[:-1]) + ("_%03d" % i)
            else:
                new_dirpath = "_".join(dirpath.split("_")[:-1]) + ("_%03d" % i)
        else:
            new_dirpath = dirpath + ("_%03d" % i)
        i += 1
    if i > 1:
        print("dirpath collision, changed to:\n%s" % new_dirpath)
    dirpath = new_dirpath
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    else:
        raise Exception("dirpath should be new, dirpath collision not corrected.")

    uu = np.zeros((4, 4))
    for i, state in enumerate(fitness_states):
        uu[:, i] = state
    criteria = lambda x, y: sum(int(xi) * 2 ** (len(x) - i - 1) for i, xi in enumerate(x)) > sum(
        int(yi) * 2 ** (len(y) - i - 1) for i, yi in enumerate(y))
    tf = fitness_states

    with open(dirpath+'/0-Notes', 'w') as notes:
        notes.write(dirpath + "\n-" * 32 + "\n" +
                    "Simulates QGA with fitness criteria based in this sorting order:\n")
        for i, state in enumerate(fitness_states):
            notes.write("%d.\n" % i)
            notes.write(repr(state))
            notes.write("\n")
        notes.write('Genetic Algorithm parameters:\n')
        notes.write('generation_numbar = {:d}\n population number = {:d}\n chromosome length = {:d}\n'.format(g, n, cl))
        notes.write('pm:\n')
        notes.write(repr(pm))
        notes.write("\n")
        notes.write('pre_projection_unitary:\n')
        notes.write(repr(ppu))
        notes.write("\n")
        notes.write('mutation_unitary:\n')
        notes.write(repr(mu))
        notes.write("\n")

    for trial in range(samples):
        print("trial ", trial, end=' ')
        #t1 = time()
        inicio = time.time()

        rho_population = qm.rho.gen_random_rho(n * cl)

        rho_final, ft = quantum_genetic_algorithm(criteria, fitness_basis=uu,
                                                  init_population=rho_population, n=n, cl=cl,
                                                  generation_number=g, pm=pm, mutation_unitary=mu,
                                                  projection_method="ptrace", pre_projection_unitary=ppu,
                                                  store_path=None,
                                                  track_fidelity=tf)
        #print(time() - t1)
        fin = time.time()
        anotar_tiempos(inicio, fin, "modulos.txt", "qga")
        with open(dirpath+'/fidelity_tracks_{:03d}'.format(trial), 'w') as file:
            file.write("Tracking fidelities for:\n")
            for i, state in enumerate(fitness_states):
                file.write("%d.\n" % i)
                file.write(repr(state))
                file.write("\n")
            file.write("\n")
            file.write(repr(ft))

if __name__ == '__main__':
    from scipy.stats import special_ortho_group
    eliminar_contenido("modulos.txt")
    comenzar = time.time()
    #state_case_number = 600
    state_case_number = 10
    #samples = 50
    samples = 4
    run_num = 9
    dirpath = 'QGA_QF_run_{:02d}/QGA_BCQO_test_'.format(run_num)

    for state_case in range(state_case_number):
        print("State case: %d" % state_case)
        if state_case == 0:
            tf = [np.array([1, 0, 0, 0]),
                np.array([0, 1, 0, 0]),
                np.array([0, 0, 1, 0]),
                np.array([0, 0, 0, 1])]
        elif state_case == 1:
            tf = [np.full(4, 1 / 2),
                np.full(4, 1 / 2) * np.array([1, -1, 1, -1]),
                np.full(4, 1 / 2) * np.array([1, 1, -1, -1]),
                np.full(4, 1 / 2) * np.array([1, -1, -1, 1])]
        else:
            # DOES NOT GENERATE COMPLEX NUMBERS!!!
            ortho_group = special_ortho_group.rvs(4)
            tf = [ortho_group[:, i] for i in range(4)]

        qga_qf_test(fitness_states=tf, samples=samples, dirpath=dirpath+("%03d" % (state_case + 1)))
    end_time = time.time()
    # Calcula e imprime el tiempo de ejecución
    terminar = time.time()
    anotar_tiempos(comenzar,terminar,"modulos.txt","Ejecutame")
    #print("Tiempo de ejecución: {:.5f} segundos".format(execution_time))