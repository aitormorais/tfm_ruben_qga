import numpy as np
import unittest
from modularizando_qga import build_unitary_routines, build_mutation_arrays,init_pop_numpy
import quantum_mats as qm
from copy import deepcopy
import time
import cProfile
import pstats
def visualizar_matrix():
        n = 2
        cl = 3
        init_population = qm.rho.gen_random_rho(n * cl)
        #init_population=[]

        # Crear un objeto rho a partir de la matriz de densidad del estado cuántico que desea rastrear
        psi = np.array([1, 0])
        density_matrix = np.outer(psi, psi.conj())  # matriz de densidad para el estado base |0>
        track_fidelity = False

        store_path = None
        generation_number = 5
        
        rho_pop= initialize_population(init_population, n, cl, track_fidelity, store_path, generation_number)
        # Asegúrese de que rho_pop es un objeto válido, fidelity_array es None y store es None
        r_pop=init_pop_numpy(init_population, n, cl, track_fidelity, store_path, generation_number)
        #print(r_pop.get_matrix())
        #print(r_pop[0].get_matrix())
        #print(r_pop[0].get(0,0))
        for numero in r_pop[1]:
            print("NUMERO: ",numero)

def verq():
    with cProfile.Profile() as pr:
        cl = 3
        n = 2
        a = n // 2
        cross_index = cl // 2
            # antes
        qq1 = [q1 for nu in range(0, n//2, 2) for q1 in range((n // 2 + nu) * cl + cross_index, (n // 2 + nu + 1) * cl)]
        qq2 = [q2 for nu in range(0, n//2, 2) for q2 in range((n // 2 + nu + 1) * cl + cross_index, (n // 2 + nu + 2) * cl)]
            #despues
        qk1 = np.array([q1 for nu in range(0, n//2, 2) for q1 in range((n // 2 + nu) * cl + cross_index, (n // 2 + nu + 1) * cl)])
        qk2 = np.array([q2 for nu in range(0, n//2, 2) for q2 in range((n // 2 + nu + 1) * cl + cross_index, (n // 2 + nu + 2) * cl)])
        print(type(qq1))
        print(type(qk1))
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    #stats.print_stats()
    stats.dump_stats(filename='needs_profiling.prof')
if __name__ == "__main__":
    #visualizar_matrix()
    
    inicio = time.time()
    verq()
    fin = time.time()
    ejecucion = fin - inicio
    print(ejecucion)
    
    
