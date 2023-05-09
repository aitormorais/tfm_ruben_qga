import numpy as np
import unittest
from modularizando_qga import *
import quantum_mats as qm
import re
from copy import deepcopy
def bucle_for(population,n,cl):
        #if n != len(population) or cl != len(population[0]):
         #   raise ValueError('n and cl mismatch. {}!={} or {}!={}.'.format(n, len(population), cl, len(population[0])))
        i = 0
        index = 0
        for p in population:
            for c in p:
                i += c * 2**(n*cl - index - 1)
                index += 1
        return i

class Testt_pop(unittest.TestCase):
    def test_tipo_matrix(self):
        cl = 3
        n = 2
        matrix = np.array([[1, 0, 1], [0, 1, 1]])
        self.assertEqual(type(qm.rho.gen_random_rho(n * cl).get_matrix()), type(matrix))
    def test_copia(self):
        init_population = np.array([[1, 0, 1], [0, 1, 1]])
        population = init_population.copy()
        pop = np.array(deepcopy(init_population))
        self.assertEqual(population.all(), pop.all())

    def test_probar_ancillas(self):
        cl = 3
        n = 2
        #init_population = qm.rho.gen_random_rho(n * cl)
        init_population = np.array([[1, 0, 1], [0, 1, 1]])
        population = deepcopy(init_population)
        pop_numpy = np.array(deepcopy(init_population))
        w=bucle_for(pop_numpy,n,cl)
        i=bucle_for(population,n,cl)
        a = n//2
        rho_pop = qm.rho([1], [i * 2**a], [i * 2**a], (2**(n*cl + a), 2**(n*cl + a)))
        r_pop = qm.rho([1], [w * 2**a], [w * 2**a], (2**(n*cl + a), 2**(n*cl + a)))
        self.assertEqual(rho_pop.get(0,0),r_pop.get(0,0))
    def test_ancillas(self):
        cl = 3
        n = 2
        a = n//2
        ancillas = [ia for ia in range(n*cl, n*cl+a)]
        # Después
        anci = np.arange(n*cl, n*cl+a)
        #print(type(ancillas[0]))
        self.assertEqual(ancillas, anci)
        self.assertEqual(len(ancillas), len(anci))
        #self.assertEqual(print(type(ancillas[0])), print(type(anci[0])))
    

    def test_initialize_population(self):
        # Prueba con una población inicial válida
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
        #for j, h in zip(r_pop, rho_pop.get_matrix()):
            #print("h: ",h[2])
            #print(j.get(1,0))
            #print(j.get(0,0))  # Para acceder a un elemento en una matriz NumPy, utiliza el índice entre corchetes.
            #print(h[0])  # h es una instancia de la clase rho, por lo que podemos usar el método get.

        self.assertEqual(rho_pop.get_matrix().all(),r_pop[0].get_matrix().all())
        
class Test_unitary_routines(unittest.TestCase):
    def test_numpy_vs_original_crossover(self):
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
        for q,k in zip(qq1,qk1):
            self.assertEqual(q, k)
        for q,k in zip(qq2,qk2):
            self.assertEqual(q, k)
        #self.assertEqual(1, "dasdewsa")
        
   # def test_numpy_vs_original_crossover_resultado(self):
        #cambiando a numpy
        #cl = 3
        #n = 2
        #a = n // 2
        #cross_index = cl // 2
        #r=crossover_rutine_unitary(n,cl,a)
        #antigua
        #qq1 = [q1 for nu in range(0, n//2, 2) for q1 in range((n // 2 + nu) * cl + cross_index, (n // 2 + nu + 1) * cl)]
        #qq2 = [q2 for nu in range(0, n//2, 2) for q2 in range((n // 2 + nu + 1) * cl + cross_index, (n // 2 + nu + 2) * cl)]
        #qq1 = np.array([q1 for nu in range(0, n//2, 2) for q1 in range((n // 2 + nu) * cl + cross_index, (n // 2 + nu + 1) * cl)])
        #qq2 = np.array([q2 for nu in range(0, n//2, 2) for q2 in range((n // 2 + nu + 1) * cl + cross_index, (n // 2 + nu + 2) * cl)])
        #cross = qm.Swap_reg[0](qq1, qq2, n*cl)
        #cross = qm.Swap_reg(qq1, qq2, n*cl)
        #cross = qm.KronExpand(1, cross, 2 ** a)
        #mat_cross = cross.get_matrix()
        #self.assertEqual(r, mat_cross)


class TestValidateMutationUnitary(unittest.TestCase):

    def test_valid_cases(self):
        valid_inputs = ["r", "R", "x", "X", "not", "NOT", "h", "H", "i", "I"]
        for input in valid_inputs:
            self.assertTrue(validate_mutation_unitary(input), f"Failed for input: {input}")

    def test_invalid_cases(self):
        invalid_inputs = ["", " ", "a", "b", "c", "Rr", "xi", "no", "NOTx", "1", "!", "ri"]
        for input in invalid_inputs:
            self.assertFalse(validate_mutation_unitary(input), f"Failed for input: {input}")


class TestMutationUnitary(unittest.TestCase):
    def test_x_unitary(self):
        valid_inputs = ["x", "X", "not", "NOT"]
        expected_result = qm.rho([1, 1], [0, 1], [1, 0], (2, 2))
        for input in valid_inputs:
            result = create_mutation_unitary(input)
            self.assertEqual(result.get_matrix().all(), expected_result.get_matrix().all())

    def test_h_unitary(self):
        valid_inputs = ["h", "H"]
        expected_result = qm.rho(np.array([1, 1, 1, -1])/np.sqrt(2), [0, 0, 1, 1], [0,  1, 0, 1], (2, 2))
        for input in valid_inputs:
            result = create_mutation_unitary(input)
            self.assertEqual(result.get_matrix().all(), expected_result.get_matrix().all())

    def test_i_unitary(self):
        valid_inputs = ["i", "I"]
        expected_result = qm.Identity(2)
        for input in valid_inputs:
            result = create_mutation_unitary(input)
            self.assertEqual(result.get_matrix().all(), expected_result.get_matrix().all())

    def test_invalid_input(self):
        invalid_inputs = ["", " ", "a", "b", "c", "Rr", "xi", "no", "1", "!", "ri"," i"]
        for input in invalid_inputs:
            with self.assertRaises(ValueError):
                create_mutation_unitary(input)

    def test_unitary_rotation(self):
        test =False
        test_str=["I","i"]
        for letra in test_str:
            if re.match(r"^[iI]$", letra):
                test=True
                self.assertEqual(True, test)
                test=False
    




if __name__ == "__main__":
    unittest.main()