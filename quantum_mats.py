"""
.. module:: quantum_mats.py
    :synopsis: Defines the Quantum_Matrix class and implements
    several specific matrices.
.. moduleauthor::  Ruben Ibarrondo (rubenibarrondo@gmail.com)
"""
from abc import ABC, abstractmethod
from math import log2, ceil, cos, sin, pi, sqrt
from random import random
import numpy as np
from scipy import linalg


class QuantumMatrix:
    """
    Clase abstracta que representa una matriz cuántica.
    """

    error_tol = 1e-5

    @abstractmethod
    def get(self, i, j):
        """
        Obtiene el elemento (i, j) de la matriz cuántica.

        :param i: Índice de fila.
        :param j: Índice de columna.
        :return: El valor del elemento (i, j) de la matriz cuántica.
        """
        if i >= self.shape[0] or j >= self.shape[1] or i < 0 or j < 0:
            raise IndexError(
                'Index out of bounds, obtained ({}, {}) for shape ({}, {})'.format(
                    i, j, self.shape[0], self.shape[1]))
        if type(i) != int or type(j) != int:
            raise IndexError(
                'Only int index allowed but {} and {} were obtained'.format(type(i), type(j))
            )
        pass

    @abstractmethod
    def row(self, i0):
        """
        Obtiene una fila de la matriz cuántica.

        :param i0: Índice de fila.
        :yield: Tuplas (j, value) que representan los elementos no nulos de la fila i0.
        """
        # [(j, value), (j, value), ...]
        if i0 >= self.shape[0] or i0 < 0:
            raise IndexError(
                'Index out of bounds, obtained ({}, :) for shape ({}, {})'.format(
                    i0, self.shape[0], self.shape[1]))
        if type(i0) != int:
            raise IndexError(
                'Only int index allowed but {} was obtained'.format(type(i0))
            )
        pass

    @abstractmethod
    def col(self, j0):
        """
        Obtiene una columna de la matriz cuántica.

        :param j0: Índice de columna.
        :yield: Tuplas (i, value) que representan los elementos no nulos de la columna j0.
        """
        # [(i, value), (i, value), ...]
        if j0 >= self.shape[1] or j0 < 0:
            raise IndexError(
                'Index out of bounds, obtained (:, {}) for shape ({}, {})'.format(
                    j0, self.shape[0], self.shape[1]))
        if type(j0) != int:
            raise IndexError(
                'Only int index allowed but {} was obtained'.format(type(j0))
            )
        pass

    @abstractmethod
    def dot(self, M):
        """
        Realiza el producto de la matriz cuántica con otra matriz cuántica.

        :param M: La matriz cuántica a multiplicar.
        """
        if self.shape != M.shape:
            raise IndexError(
                'Shape mismatch, obtained ({}, {}) and ({}, {})'.format(
                    self.shape[0], self.shape[1], M.shape[0], M.shape[1]))
        pass

    def print_mat(self, block=None, len_lim=32):
        """
        Imprime la matriz cuántica.

        :param block: Número de elementos por bloque en la impresión de la matriz (opcional).
        :param len_lim: Límite de longitud para imprimir la matriz (opcional).
        """
        if self.shape[0] > len_lim:
            raise ValueError('Too big for printing. shape = {}'.format(self.shape))
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                print(self.get(i, j), end=' ')
                if block and j%block == block-1:
                    print(end=' ')
            print()
            if block and i%block == block - 1:
                print()

    def get_matrix(self, len_lim=2**12):
        """
        Devuelve una matriz NumPy correspondiente a la matriz cuántica actual. Los elementos son obtenidos a partir de
        los elementos individuales de la matriz utilizando el método 'row'.

        :param len_lim: límite máximo para la longitud de la matriz devuelta.
        :return: una matriz NumPy correspondiente a la matriz cuántica actual.
        :raises ValueError: si la longitud de la matriz cuántica excede el valor de 'len_lim'.
        """
        if self.shape[0] > len_lim:
            raise ValueError('Too big for getting matrix with len_lim={}. shape = {}'.format(len_lim, self.shape))
        mat = np.zeros(self.shape)
        for i in range(self.shape[0]):
            for j, v in self.row(i):
                mat[i, j] = v
        return mat


# -------------------
#
#    Main matrices
#
# -------------------


class rho(QuantumMatrix):
    """
    Clase que representa una matriz de densidad cuántica.
    Hereda de la clase QuantumMatrix.
    """
    def __init__(self, data, ii=None, jj=None,  shape=None, dense=False):
        """
        Inicializa la instancia de la clase rho.
        
        :param data: Los datos de la matriz de densidad.
        :param ii: Índices de fila.
        :param jj: Índices de columna.
        :param shape: La forma de la matriz.
        :param dense: Indicador de si la matriz es densa (True) o dispersa (False).
        :raises ValueError: Si la matriz de densidad no cumple las propiedades requeridas.
        """

        self.dense = dense
        if not dense:
            self.shape = shape
            self.store = {(i, j): d for d, i, j in zip(data, ii, jj)}
            self.nonzeros = len(self.store)
            self.nq = ceil(log2(self.shape[0]))
        if dense:
            """se realizan comprobaciones para garantizar que la matriz cumple las propiedades requeridas para una matriz de densidad:
            1-La traza de la matriz debe ser igual a 1 (o cercana a 1, dentro de un cierto límite de error). la traza se refiere a la suma de los elementos diagonales
            2-La matriz debe ser hermítica, es decir, igual a su traspuesta conjugada.
            3-La matriz debe ser positiva semidefinida, lo que significa que todos sus valores propios son no negativos.
            Si alguna de estas comprobaciones falla, se lanza una excepción ValueError. Si la matriz cumple las condiciones,
            se almacenan su forma (self.shape),
            sus datos (self.store),
            el número de elementos no nulos (self.nonzeros) y el número de qubits (self.nq)."""
            if (abs(np.trace(data) - 1) > QuantumMatrix.error_tol).all():
                raise ValueError("rho not normalized, tr: ", abs(np.trace(data) - 1))
            data = np.array(data)
            if np.any(abs(np.transpose(data.conjugate()) - data) > QuantumMatrix.error_tol):#verifica si algún elemento de la diferencia calculada en el paso 3 es mayor que un cierto umbral de tolerancia de erro
                #data.conjugate() ->  calcula la conjugada  de data
                # abs(np.transpose(data.conjugate()) - data) ->calcula el valor absoluto de la diferencia entre la matriz transpuesta conjugada y la matriz original data.
                # Si la matriz es hermítica, esta diferencia debe ser muy pequeña para todos los elementos de la matriz.
                raise ValueError("rho not Hermitian, max dev: ", np.max(abs(np.transpose(data.conjugate()) - data)))#np.transpose()calcula la traspuesta de la conjugada
            if np.any(np.linalg.eigvals(data) < -QuantumMatrix.error_tol):#
                raise ValueError("rho not positive define, min val: ", np.min(np.linalg.eigvals(data)))
            self.shape = data.shape
            self.store = data
            self.nonzeros = data.shape[0] * data.shape[1]
            self.nq = ceil(log2(self.shape[0]))

    def __binenc(self, i):
        """
        Convierte un número entero en su representación binaria de longitud nq.
        
        :param i: Número entero a convertir.
        :return: La representación binaria del número entero.
        """
        ib = bin(i)[2:]
        if len(ib) < self.nq:
            ib = '0' * (self.nq - len(ib)) + ib
        return ib

    def __bindec(self, ib):
        """
        Convierte una representación binaria en su número entero equivalente.
        
        :param ib: Representación binaria.
        :return: El número entero correspondiente a la representación binaria.
        """
        return sum(2 ** (len(ib) - j - 1) * int(ib[j]) for j in range(len(ib)))

    def __add__(self, other):
        """
        Suma de dos matrices de densidad rho.
        
        :param other: Otra matriz de densidad rho.
        :return: La matriz resultante de la suma.
        :raises NotImplementedError: Si la operación no está implementada para el tipo de matriz dado.
        """
        QuantumMatrix.dot(self, other)
        if type(other) == np.ndarray and self.dense:
            return self.store + other
        elif type(other) != rho:
            raise NotImplementedError('__add__ is only implemented for rho matrices.')

        new_rho = rho([], [], [], self.shape)
        for i, j in self.store.keys():
            new_rho.set(i, j, self.get(i, j))

        for i, j in other.store.keys():
            if new_rho.get(i, j) == 0:
                new_rho.set(i, j, other.get(i, j))
            else:
                new_rho.set(i, j, new_rho.get(i, j) + other.get(i, j))
        return new_rho

    def __sub__(self, other):
        """
        Resta dos matrices de densidad rho y devuelve la matriz resultante.
        
        :param other: Otra matriz de densidad rho.
        :return: La matriz resultante de la resta.
        :raises NotImplementedError: Si la operación no está implementada para el tipo de matriz dado.
        """
        QuantumMatrix.dot(self, other)
        if type(other) == np.ndarray and self.dense:
            return self.store - other
        elif type(other) != rho:
            raise NotImplementedError('__sub__ is only implemented for rho matrices.')

        new_rho = rho([], [], [], self.shape)
        for i, j in self.store.keys():
            new_rho.set(i, j, self.get(i, j))

        for i, j in other.store.keys():
            if new_rho.get(i, j) == 0:
                new_rho.set(i, j, -other.get(i, j))
            else:
                new_rho.set(i, j, new_rho.get(i, j) - other.get(i, j))
        return new_rho

    def __mul__(self, other):
        """
        Multiplica la matriz de densidad rho por un número escalar y devuelve la nueva matriz rho resultante.
        
        :param other: Número escalar (int o float).
        :return: La matriz resultante de la multiplicación.
        :raises NotImplementedError: Si la operación no está implementada para el tipo de matriz dado.
        """
        if (type(other) != int or type(other) != float) and self.dense:
            return self.store * other
        elif type(other) != int and type(other) != float:
            raise NotImplementedError('__mul__ is only implemented for rho with int and float, but {} was obtained'.format(type(other)))
        new_rho = rho([], [], [], self.shape)
        for i, j in self.store.keys():
            new_rho.set(i, j, other * self.get(i, j))
        return new_rho

    def __truediv__(self, other):
        """
        Divide la matriz de densidad rho por un número escalar y devuelve la nueva matriz rho resultante.
        
        :param other: Número escalar (int o float).
        :return: La matriz resultante de la división.
        :raises NotImplementedError: Si la operación no está implementada para el tipo de matriz dado.
        """
        if (type(other) != int or type(other) != float) and self.dense:
            return self.store / other
        elif type(other) != int and type(other) != float:
            raise NotImplementedError('__div__ is only implemented for rho with int and float, but {} was obtained'.format(type(other)))
        new_rho = rho([], [], [], self.shape)
        for i, j in self.store.keys():
            new_rho.set(i, j, self.get(i, j)/other)
        return new_rho

    def get_matrix(self, len_lim=2**12):
        """
        Obtiene la representación de la matriz de densidad rho en forma de una matriz numpy.
        
        :param len_lim: Límite de longitud para la matriz.
        :return: La matriz numpy que representa la matriz de densidad.
        :raises ValueError: Si la matriz es demasiado grande para obtenerla con el límite de longitud dado.
        """
        if not self.dense:
            return QuantumMatrix.get_matrix(self, len_lim)
        else:
            return self.store

    @staticmethod
    def gen_rho_from_state(coefficients, state_indices, length):
        """
        Genera una matriz rho a partir de los coeficientes y los índices de estado proporcionados.

        :param coefficients: Lista de coeficientes.
        :param state_indices: Lista de índices de estado.
        :param length: Longitud de la matriz cuadrada resultante.
        :return: Una instancia de la clase rho construida a partir de los coeficientes y los índices de estado.
        """
        r = rho([], [], [], (length, length))

        if len(coefficients) != len(state_indices):
            raise ValueError('coefficients and state_indices must have same lenght')

        for p in range(len(state_indices)):
            for q in range(len(state_indices)):
                r.set(state_indices[p], state_indices[q], coefficients[p]*coefficients[q])
        return r
    @staticmethod
    def gen_rho_from_matrix(matrix):
        """
        Genera una matriz rho a partir de una matriz densa dada.

        :param matrix: La matriz densa a partir de la cual se generará la matriz rho.
        :return: Una instancia de la clase rho construida a partir de la matriz proporcionada.
        """
        return rho(matrix, dense=True)

    @staticmethod
    def gen_random_rho(nq, dense=True, constraints={"pure": True, "separable": False}, asvector=False):
        """
        Generate a random rho state for nq qubits.
        Only dense=True and constraints["pure"] = True are supported yet.
        The random pure state generator is based on the Qiskit random_statevector function (bases in Haar measures).
        :param nq:
        :param dense:
        :param constraints:
        :param asvector:
        :return:
        """
        if not dense and not constraints.get("pure", False):
            raise ValueError("Only dense=True and constraints[\"pure\"] = True are supported yet.")
        if constraints.get("separable", False):
            raise ValueError("Constraints[\"separable\"] = True is not supported yet.")

        dim = 2**nq
        x = np.random.random(dim)
        x += x == 0
        x = -np.log(x)
        sumx = sum(x)
        phases = np.random.random(dim) * 2.0 * np.pi

        statevector = np.sqrt(x / sumx) * np.exp(1j * phases)
        if asvector and constraints.get("pure", True):
            return statevector
        else:
            mat = np.kron(statevector.reshape((statevector.shape[0], 1)), statevector.conjugate())
            return rho.gen_rho_from_matrix(mat)

    @staticmethod
    def get_xx_basis():
        """
    Genera las cuatro bases de Bell xx (pp, pm, mp, mm).
    :return: Las cuatro bases de Bell xx (pp, pm, mp, mm).
        """
        pp = rho.gen_rho_from_state(np.full(4, 1 / 2), range(4), 4)
        pm = rho.gen_rho_from_state(np.full(4, 1 / 2) * np.array([1, -1, 1, -1]), range(4), 4)
        mp = rho.gen_rho_from_state(np.full(4, 1 / 2) * np.array([1, 1, -1, -1]), range(4), 4)
        mm = rho.gen_rho_from_state(np.full(4, 1 / 2) * np.array([1, -1, -1, 1]), range(4), 4)
        return pp, pm, mp, mm

    def str_qubit_state(self, block=None):
        state = ''
        for i, j in self.store:
            ib = self.__binenc(i)
            jb = self.__binenc(j)
            if not block:
                if state != '':
                    state += '\n'
                state += ib + ", " + jb + ': ' + str(self.get(i, j))
            if block:
                ibin = ''
                for k, b in enumerate(ib):
                    if k % block == 0 and k != 0:
                        ibin += ' ' + b
                    else:
                        ibin += b
                jbin = ''
                for k, b in enumerate(jb):
                    if k % block == 0 and k != 0:
                        jbin += ' ' + b
                    else:
                        jbin += b
                if state != '':
                    state += '\n'
                state += ibin + ", " + jbin + ': ' + str(self.get(i, j))
        return state

    def get(self, i, j):
        """
        Obtiene el elemento (i, j) de la matriz rho.
        :param i: Índice de fila.
        :param j: Índice de columna.
        :return: El elemento (i, j) de la matriz rho.
        """
        QuantumMatrix.get(self, i, j)
        if not self.dense:
            return self.store.get((i, j), 0)
        else:
            return self.store[i, j]

    def set(self, i, j, value):
        """
    Establece el valor del elemento (i, j) en la matriz rho.
    :param i: Índice de fila.
    :param j: Índice de columna.
    :param value: Valor a establecer en la posición (i, j).
    """
        QuantumMatrix.get(self, i, j)
        # This method should also check that this matrix
        # fulfils the constrains of a state matrix.
        if not self.dense:
            if value != 0:
                self.store[(i, j)] = value
            elif (i, j) in self.store.keys():
                self.store.pop((i, j))
            self.nonzeros = len(self.store)
        else:
            self.store[i, j] = value

    def col(self, j0):
        """
        devuelve un iterador sobre los elementos de la columna j0 de la matriz.
        :param j0: Índice de la columna.
        :return: Un iterador sobre los elementos de la columna j0.


        """
        QuantumMatrix.col(self, j0)
        if not self.dense:
            for i, j in self.store.keys():
                if j == j0:
                    yield i, self.store[(i, j)]
        else:
            return self.store[:, j0]

    def row(self, i0):
        """
        devuelve un iterador sobre los elementos de la fila i0 de la matriz
        :param i0: Índice de la fila.
        :return: Un iterador sobre los elementos de la fila i0.


        """
        QuantumMatrix.row(self, i0)
        if not self.dense:
            for i, j in self.store.keys():
                if i == i0:
                    yield j, self.store[(i, j)]
        else:
            return self.store[i0, :]

    def trace(self):
        """calcula la traza de la matriz.
        :return: El valor de la traza de la matriz.
        """
        if not self.dense:
            tr = 0
            for i, j in self.store.keys():
                if i == j:
                    tr += self.get(i, j)
            return tr
        else:
            return self.store.trace()

    def partial_trace(self, qubit_2_keep, len_lim=2**12):
        """ Calculate the partial trace for qubit system.
        self is casted to a matrix.
        From neversakura/parital_trace.py
        https://gist.github.com/neversakura/d6a60b4bb2990d252e9e89e5629d5553#file-parital_trace-py
        Parameters
        ----------
        qubit_2_keep: list
            Index of qubit to be kept after taking the trace
        len_lim: int
            Maximum length allowed for the state. Defaults to 2**12.
        Returns
        -------
        rho_res: rho
            Density matrix after taking partial trace, type dense
        """
        r = self.get_matrix(len_lim)
        num_qubit = int(np.log2(r.shape[0]))
        qubit_axis = [(i, num_qubit + i) for i in range(num_qubit)
                      if i not in qubit_2_keep]
        minus_factor = [(i, 2 * i) for i in range(len(qubit_axis))]
        minus_qubit_axis = [(q[0] - m[0], q[1] - m[1])
                            for q, m in zip(qubit_axis, minus_factor)]
        rho_res = np.reshape(r, [2, 2] * num_qubit)
        qubit_left = num_qubit - len(qubit_axis)
        for i, j in minus_qubit_axis:
            rho_res = np.trace(rho_res, axis1=i, axis2=j)
        if qubit_left > 1:
            rho_res = np.reshape(rho_res, [2 ** qubit_left] * 2)
        return rho(rho_res, dense=True)

    def dot(self, M):
        """Calcula el producto matricial entre la matriz y M
        :param M: La matriz con la que se va a calcular el producto matricial.
        :return: La matriz resultante del producto matricial.
        """
        QuantumMatrix.dot(self, M)
        if type(M) == np.ndarray and self.dense:
            return np.dot(self.store, M)
        elif type(M) != np.ndarray and self.dense:
            raise NotImplementedError('dot product for dense state only supports np.ndarray.')
        # assuming rho has smaller nonzero number
        # add shape checking
        new_rho = rho([], [], [], self.shape)
        for i, k in self.store.keys():
            for j, v in M.row(k):
                if new_rho.get(i, j) == 0:
                    new_rho.set(i, j, v * self.get(i, k))
                else:
                    new_rho.set(i, j, new_rho.get(i, j) + v * self.get(i, k))
        
        return new_rho

    def set_qubit(self, q, v, measure_it='random'):
        '''
        This method can be used to set a qubit to 1 or 0.
        The process can happen in 2 ways:
            - Measure the qubit and get v
            - Measure the qubit and get !v, perform X in the qubit
        Both processes can not happen simultaneously in the same state.
        For an state that has only one value for the qubit this means nothing,
        but if there is a superposition loss of information may happen.

        If the parameter measure_it is set to 'v', the default option will be to directly measure
        the value 'v' in the register, only measuring !v if the previous state doesn't exist.
        If set to '!v' the other option will be priorized.
        If set to 'random', the intrinsic probabilities of the states will be used to choose the case.
        '''
        if v != 0 and v != 1:
            raise ValueError('Qubits can only be set to 0 or 1, but {} was obtained.'.format(v))
        if self.trace() == 0:
            raise ValueError('The rho state should have trace = 1.')

        if measure_it == 'v':
            measure_it = True
        elif measure_it == '!v':
            measure_it = False
        elif measure_it == 'random':
            r = random()
            p = 0
            for i, j in self.store.keys():
                if i == j:
                    if self.__binenc(i)[q] == str(v):
                        p += self.get(i, i)
            measure_it = r < p
        else:
            raise ValueError(
                "measure_it option can only be set to 'v', '!v' or 'random', but {} was obtained.".format(measure_it))

        if measure_it:
            new_rho = rho([], [], [], self.shape)
            for i, j in self.store.keys():
                ib = self.__binenc(i)
                jb = self.__binenc(j)
                if ib[q] == str(v) and jb[q] == str(v):
                    new_rho.set(i, j, self.get(i, j))
                else:
                    new_rho.set(i, j, 0)
            tr = new_rho.trace()
            if tr != 0:
                new_rho = new_rho / tr
                self.store = new_rho.store
                self.nonzeros = new_rho.nonzeros
            else:
                measure_it = False

        if not measure_it:
            new_rho = rho([], [], [], self.shape)
            for i, j in self.store.keys():
                ib = self.__binenc(i)
                jb = self.__binenc(j)
                if ib[q] != str(v) and jb[q] != str(v):
                    if q < self.nq-1 and q > 0:
                        inew = self.__bindec(ib[:q]+str(v)+ib[q+1:])
                        jnew = self.__bindec(jb[:q]+str(v)+jb[q+1:])
                    elif q == 0:
                        inew = self.__bindec(str(v) + ib[q + 1:])
                        jnew = self.__bindec(str(v) + jb[q + 1:])
                    elif q == self.nq-1:
                        inew = self.__bindec(ib[:q] + str(v))
                        jnew = self.__bindec(jb[:q] + str(v))

                    new_rho.set(inew, jnew, self.get(i, j))
                    new_rho.set(i, j, 0)
                else:
                    new_rho.set(i, j, 0)
            tr = new_rho.trace()
            if tr != 0:
                new_rho = new_rho / tr
                self.store = new_rho.store
                self.nonzeros = new_rho.nonzeros
            else:
                self.set_qubit(q, v, 'v')

    def projection_controlled_rotation(self, q, U0=1, U1=1, pre_projection_unitary=1, projection_method='r'):
        '''
        1-Notes
            - Check Hermitian Conjugate is working.
        :q: the qubit to apply the controlled rotation
        :U0: the unitary to be applied if 0 is obtained, default 1 for identity
        :U1: the unitary to be applied if 1 is obtained, default 1 for identity
        :pre_projection_unitary: the unitary to be applied before the projection is performed, default 1 for identity
        :projection_method: defines the way the projection is performed. If set to 'r', it computes the probability to
         measure each possibility and selects according to those weights. If set to 0 or 1, it prioritizes measuring that
         option first.
        '''

        if self.trace() == 0:
            raise ValueError('The rho state should have trace = 1.')

        if pre_projection_unitary != 1:
            new_rho = KronExpand(2 ** q, pre_projection_unitary, 2 ** (self.nq - 1 - q)).dot(
                self).dot(HermitianConjugate(KronExpand(2 ** q, pre_projection_unitary, 2 ** (self.nq - 1 - q))))
            self.store = new_rho.store
            self.nonzeros = new_rho.nonzeros

        if projection_method == 'r':
            p = 0
            for i, j in self.store.keys():
                if i == j:
                    if self.__binenc(i)[q] == '0':
                        p += self.get(i, i)
            measure_0 = random() < p
        elif projection_method == 0:
            measure_0 = True
        elif projection_method == 1:
            measure_0 = False
        else:
            raise ValueError('Parameter projection_method has to be either 0, 1 or \'r\'.')

        if measure_0:
            new_rho = rho([], [], [], self.shape)
            for i, j in self.store.keys():
                ib = self.__binenc(i)
                jb = self.__binenc(j)
                if ib[q] == '0' and jb[q] == '0':
                    new_rho.set(i, j, self.get(i, j))
                else:
                    new_rho.set(i, j, 0)
            tr = new_rho.trace()
            if tr != 0:
                new_rho = new_rho / tr
                self.store = new_rho.store
                self.nonzeros = new_rho.nonzeros
            else:
                raise ValueError('ctrl_rotation resulted in trace = 0.')

            if U0 != 1:
                new_rho = KronExpand(2 ** q, U0, 2 ** (self.nq - q - 1)).dot(self).dot(
                    HermitianConjugate(KronExpand(2 ** q, U0, 2 ** (self.nq - q - 1))))
                self.store = new_rho.store
                self.nonzeros = new_rho.nonzeros

        if not measure_0:
            new_rho = rho([], [], [], self.shape)
            for i, j in self.store.keys():
                ib = self.__binenc(i)
                jb = self.__binenc(j)
                if ib[q] == '1' and jb[q] == '1':
                    new_rho.set(i, j, self.get(i, j))
                else:
                    new_rho.set(i, j, 0)
            tr = new_rho.trace()
            if tr != 0:
                new_rho = new_rho / tr
                self.store = new_rho.store
                self.nonzeros = new_rho.nonzeros
            else:
                raise ValueError('ctrl_rotation resulted in trace = 0.')

            if U1 != 1:
                new_rho = KronExpand(2 ** q, U1, 2 ** (self.nq - q - 1)).dot(self).dot(
                    HermitianConjugate(KronExpand(2 ** q, U1, 2 ** (self.nq - q - 1))))
                self.store = new_rho.store
                self.nonzeros = new_rho.nonzeros

    def fidelity(self, other, len_lim=2**12):
        """
        Fidelity between a density matrix and a pure state/other density matrix.
        The state is casted to a np matrix.
            F(r, s) = tr^2 √(√r s √r).
        Parameters
        ----------
        other: np.ndarray, qm.rho
            Vector representation of a pure state, or qm.rho object representing other density matrix.
        len_lim: int
            Maximum length allowed for the state. Defaults to 2**12.
        Returns
        -------
        fidelity: float
            The fidelity between pure and rho
        """
        r = self.get_matrix(len_lim)
        if type(other) == np.ndarray:
            pure = other
            if abs(np.dot(pure.conjugate(), pure) - 1) > QuantumMatrix.error_tol:
                raise ValueError("pure not normalized")
            if abs(np.trace(r) - 1) > QuantumMatrix.error_tol:
                raise ValueError("rho not normalized")
            if np.any(abs(np.transpose(r.conjugate()) - r) > QuantumMatrix.error_tol):
                raise ValueError("rho not Hermitian")
            if np.any(np.linalg.eigvals(r) < -QuantumMatrix.error_tol):
                raise ValueError("rho not positive define")

            fidelity = np.dot(pure.transpose().conjugate(), np.dot(r, pure)).real
        elif type(other) == type(self):
            r2sqrt = linalg.sqrtm(other.get_matrix(len_lim))
            fidelity = np.trace(linalg.sqrtm(np.dot(r2sqrt, np.dot(r, r2sqrt)))) ** 2
        else:
            raise ValueError("other with type %s not supported." % str(type(other)))

        if fidelity.real < -QuantumMatrix.error_tol:
            raise Exception("Negative value obtained for fidelity")
        elif abs(fidelity.imag) > QuantumMatrix.error_tol:
            raise Exception("Complex value obtained for fidelity.")
        elif abs(fidelity) > 1 + QuantumMatrix.error_tol:
            raise Exception("Obtained fidelity greater than 1.")
        else:
            return min(abs(fidelity), 1)

    def get_expected_value(self, observable):
        """
        Returns the expected value of the observable.
        Both objects must be described in the same basis.

        :param observable: np.ndarray describing the matrix of the observable.
        :return:
        """
        return np.trace(np.dot(observable, self.get_matrix()))

    def entropy(self):
        """
        Computes the entropy of the density matrix.

        :return: The value of the entropy of the state.
        """
        weights = np.ma.masked_equal(np.linalg.eigvals(self.get_matrix()), 0)
        s = - np.sum(weights * np.log2(weights))
        if s.real < -QuantumMatrix.error_tol:
            raise Exception("Negative value obtained for entropy.")
        elif abs(s.imag) > QuantumMatrix.error_tol:
            raise Exception("Complex value obtained for entropy.")
        return s.real


class Uclone(QuantumMatrix):
    """
    Clase que representa una matriz de clonación cuántica U_n de tamaño nxn.

    Atributos:
    ----------
    n: int
        Tamaño de la matriz de clonación cuántica U_n.

    shape: tuple
        Tupla que contiene el tamaño de la matriz (n*n, n*n).

    nonzeros: int
        Número de elementos no nulos en la matriz.

    Métodos:
    -------
    get(i, j):
        Retorna el valor del elemento (i, j) de la matriz U_n.

    col(j0):
        Retorna un iterador que contiene los valores no nulos en la columna j0 de la matriz U_n.

    row(i0):
        Retorna un iterador que contiene los valores no nulos en la fila i0 de la matriz U_n.

    dot(M):
        Realiza el producto matricial entre la matriz U_n y la matriz M. Retorna la matriz resultante.

    """

    def __init__(self, n):
        """
        Inicializa un objeto Uclone.
        Parámetros:
        ----------
        n: int
            Tamaño de la matriz de clonación cuántica U_n.
        """
        self.n = n
        self.shape = (n * n, n * n)
        self.nonzeros = n

    def get(self, i, j):
        """
        Retorna el valor del elemento (i, j) de la matriz U_n.
        Parámetros:
        ----------
        i: int
            Índice de fila del elemento.

        j: int
            Índice de columna del elemento.

        Returns:
        -------
        value: float
            Valor del elemento (i, j).

        """
        QuantumMatrix.get(self, i, j)
        if i // self.n == j // self.n:
            nu = i // self.n
            ip = i % self.n
            jp = j % self.n
            if jp <= self.n - nu and ip == (jp + nu):
                return 1
            elif ip == (jp + nu - self.n):
                return 1
            else:
                return 0
        else:
            return 0

    def col(self, j0):
        """
        Retorna un iterador que contiene los valores no nulos en la columna j0 de la matriz U_n.

        Parámetros:
        ----------
        j0: int
            Índice de columna.

        Yields:
        ------
        (i, value): tuple
            Tupla que contiene el índice de fila y el valor no nulo correspondiente en la columna j0.

        """
        QuantumMatrix.col(self, j0)
        nu = j0 // self.n
        jp = j0 % self.n
        if jp <= self.n - nu - 1:
            yield nu * self.n + jp + nu, 1
        else:
            yield nu * self.n + jp + nu - self.n, 1

    def row(self, i0):
        """
        Retorna un iterador que contiene los valores no nulos en la fila i0 de la matriz U_n.

        Parámetros:
        ----------
        i0: int
            Índice de fila.

        Yields:
        ------
        (j, value): tuple
            Tupla que contiene el índice de columna y el valor no nulo correspondiente en la fila i0.

        """
        QuantumMatrix.row(self, i0)
        nu = i0 // self.n
        ip = i0 % self.n
        if ip - nu >= 0:
            yield i0 - nu, 1
        else:
            yield i0 - nu + self.n, 1

    def dot(self, M):
        """
    Calcula el producto punto de la matriz cuadrada y otra matriz.
    :param M: matriz a multiplicar.
    :return: matriz resultante del producto punto.
    """
        QuantumMatrix.dot(self, M)
        if type(M) == rho:
            new_rho = rho([], [], [], self.shape)
            for k, j in M.store.keys():
                for i, v in self.col(k):
                    if new_rho.get(i, j) == 0:
                        new_rho.set(i, j, v * M.get(k, j))
                    else:
                        new_rho.set(i, j, new_rho.get(i, j) + v * M.get(k, j))
            return new_rho
        else:
            return Dot(self, M)


class Swap(QuantumMatrix):

    """
    Representa la matriz de intercambio para un sistema de dos qubits.

    Atributos:
        q1 (int): Índice del primer qubit a intercambiar.
        q2 (int): Índice del segundo qubit a intercambiar.
        nq (int): Número total de qubits en el sistema.
        shape (tuple): Dimensiones de la matriz.
        nonzeros (int): Número de elementos distintos de cero en la matriz.
    """

    def __init__(self, q1, q2, nq):
        """
        Inicializa una matriz de intercambio cuántico entre dos qubits.
        :param q1: índice del primer qubit a intercambiar.
        :param q2: índice del segundo qubit a intercambiar.
        :param nq: número total de qubits del sistema.
        """
        self.nq = nq
        self.q1 = q1
        self.q2 = q2
        self.shape = (2 ** nq, 2 ** nq)
        self.nonzeros = 2 ** nq

    def __binenc(self, i):
        """
        Convierte un índice decimal en su representación binaria con un ancho fijo.
        :param i: índice decimal a convertir.
        :return: cadena de caracteres que representa el índice en binario.
        """
        ib = bin(i)[2:]
        if len(ib) < self.nq:
            ib = '0' * (self.nq - len(ib)) + ib
        return ib

    def __bindec(self, ib):
        """
        Convierte una cadena de caracteres que representa un índice binario en su valor decimal.

        :param ib: cadena de caracteres que representa el índice en binario.
        :return: valor decimal del índice.
        """
        return sum(2 ** (len(ib) - j - 1) * int(ib[j]) for j in range(len(ib)))

    def get(self, i, j):
        """
        Devuelve el valor de la posición (i, j) de la matriz.

        :param i: índice de fila.
        :param j: índice de columna.
        :return: valor de la posición (i, j) de la matriz.
        """
        QuantumMatrix.get(self, i, j)
        ibin = self.__binenc(i)
        jbin = self.__binenc(j)
        for q in range(self.nq):
            if q in [self.q1, self.q2]:
                continue
            if ibin[q] != jbin[q]:
                return 0
        if ibin[self.q1] == jbin[self.q2] and ibin[self.q2] == jbin[self.q1]:
            return 1
        else:
            return 0

    def col(self, j0):
        """
        Devuelve un iterador sobre las entradas no nulas de la columna j0 de la matriz.

        :param j0: índice de la columna.
        :yield: tupla (i, valor) de cada entrada no nula de la columna j0.
        """
        QuantumMatrix.col(self, j0)
        jbin = self.__binenc(j0)
        ibin = ''
        for q in range(self.nq):
            if q == self.q1:
                ibin += jbin[self.q2]
            elif q == self.q2:
                ibin += jbin[self.q1]
            else:
                ibin += jbin[q]
        yield self.__bindec(ibin), 1

    def row(self, i0):
        """
    Devuelve un iterador sobre la fila i0 de la matriz de swap.
    :param i0: índice de la fila a devolver.
    :return: iterador que devuelve tuplas (índice de columna, valor en la posición (i0, columna)).
    """
        QuantumMatrix.row(self, i0)
        ibin = self.__binenc(i0)
        jbin = ''
        for q in range(self.nq):
            if q == self.q1:
                jbin += ibin[self.q2]
            elif q == self.q2:
                jbin += ibin[self.q1]
            else:
                jbin += ibin[q]
        yield self.__bindec(jbin), 1

    def dot(self, M):
        """
    Calcula el producto punto de la matriz cuadrada y otra matriz.
    :param M: matriz a multiplicar.
    :return: matriz resultante del producto punto.
    """
        QuantumMatrix.dot(self, M)
        if type(M) == rho:
            new_rho = rho([], [], [], self.shape)
            for k, j in M.store.keys():
                for i, v in self.col(k):
                    if new_rho.get(i, j) == 0:
                        new_rho.set(i, j, v * M.get(k, j))
                    else:
                        new_rho.set(i, j, new_rho.get(i, j) + v * M.get(k, j))
            return new_rho
        else:
            return Dot(self, M)


class Swap_reg(QuantumMatrix):

    def __init__(self, qq1, qq2, nq):
        self.nq = nq
        self.qq1 = qq1
        self.qq2 = qq2
        self.shape = (2 ** nq, 2 ** nq)
        self.nonzeros = 2 ** nq

    def __binenc(self, i):
        ib = bin(i)[2:]
        if len(ib) < self.nq:
            ib = '0' * (self.nq - len(ib)) + ib
        return ib

    def __bindec(self, ib):
        return sum(2 ** (len(ib) - j - 1) * int(ib[j]) for j in range(len(ib)))

    def get(self, i, j):
        QuantumMatrix.get(self, i, j)
        ibin = self.__binenc(i)
        jbin = self.__binenc(j)
        for q in range(self.nq):
            if (q in self.qq1) or (q in self.qq2):
                continue
            if ibin[q] != jbin[q]:
                return 0
        if all(ibin[q1] == jbin[q2] and ibin[q2] == jbin[q1] for q1, q2 in zip(self.qq1, self.qq2)):
            return 1
        else:
            return 0

    def col(self, j0):
        QuantumMatrix.col(self, j0)
        jbin = self.__binenc(j0)
        ibin = [b for b in jbin]
        for q1, q2 in zip(self.qq1, self.qq2):
            temp = ibin[q1]
            ibin[q1] = ibin[q2]
            ibin[q2] = temp
        yield self.__bindec(''.join(ibin)), 1

    def row(self, i0):
        QuantumMatrix.row(self, i0)
        ibin = self.__binenc(i0)
        jbin = [b for b in ibin]
        jbin = np.array(jbin)
        #print(type(jbin))
        #print(jbin)
        #print("QQ1",self.qq1)
        #print("QQ1",self.qq2)
        #print(type(self.qq1))
        for q1, q2 in zip(self.qq1, self.qq2):
            #index = np.where(qq1=q1)
            temp = jbin[q1]
            #temp = jbin[index]
            #print("q1:", q1)
            #print("q2:", q2)
            #print(type(self.qq1))
            jbin[q1] = jbin[q2]
            jbin[q2] = temp
        yield self.__bindec(''.join(jbin)), 1



    def dot(self, M):
        QuantumMatrix.dot(self, M)
        if type(M) == rho:
            new_rho = rho([], [], [], self.shape)
            for k, j in M.store.keys():
                for i, v in self.col(k):
                    if new_rho.get(i, j) == 0:
                        new_rho.set(i, j, v * M.get(k, j))
                    else:
                        new_rho.set(i, j, new_rho.get(i, j) + v * M.get(k, j))
            return new_rho
        else:
            return Dot(self, M)


class CSwap(QuantumMatrix):
    """
    Clase que representa una matriz de intercambio entre dos registros en un sistema cuántico.
    Hereda de QuantumMatrix.
    """

    def __init__(self, c, q1, q2, nq):
        """
        Inicializa una instancia de Swap_reg.
        :param qq1: Lista de índices del primer registro a intercambiar.
        :param qq2: Lista de índices del segundo registro a intercambiar.
        :param nq: Número total de qubits en el sistema.
        """
        self.nq = nq
        self.c = c
        self.q1 = q1
        self.q2 = q2
        self.shape = (2 ** nq, 2 ** nq)
        self.nonzeros = 2 ** nq

    def __binenc(self, i):
        """
        Convierte un entero a su representación binaria como cadena de caracteres.
        :param i: Entero a convertir.
        :return: Representación binaria del entero.
        """
        ib = bin(i)[2:]
        if len(ib) < self.nq:
            ib = '0' * (self.nq - len(ib)) + ib
        return ib

    def __bindec(self, ib):
        """
        Convierte una cadena de caracteres que representa un número binario a su equivalente entero.
        :param ib: Cadena de caracteres que representa un número binario.
        :return: Entero equivalente al número binario.
        """
        return sum(2 ** (len(ib) - j - 1) * int(ib[j]) for j in range(len(ib)))

    def get(self, i, j):
        """
        Obtiene el valor en la posición (i, j) de la matriz de intercambio.
        :param i: Índice de fila.
        :param j: Índice de columna.
        :return: Valor en la posición (i, j).
        """
        QuantumMatrix.get(self, i, j)
        ibin = self.__binenc(i)
        jbin = self.__binenc(j)
        if ibin[self.c] != jbin[self.c]:
            return 0
        if ibin[self.c] == '0':
            if i == j:
                return 1
            else:
                return 0
        else:
            for q in range(self.nq):
                if q in [self.q1, self.q2]:
                    continue
                if ibin[q] != jbin[q]:
                    return 0
            if ibin[self.q1] == jbin[self.q2] and ibin[self.q2] == jbin[self.q1]:
                return 1
            else:
                return 0

    def col(self, j0):
        """
        Devuelve un iterador sobre la columna j0 de la matriz de intercambio.
        :param j0: Índice de columna.
        :return: Iterador que devuelve tuplas (índice de fila, valor en la posición (fila, j0)).
        """
        QuantumMatrix.col(self, j0)
        jbin = self.__binenc(j0)
        if jbin[self.c] == '0':
            yield j0, 1
        else:
            ibin = ''
            for q in range(self.nq):
                if q == self.q1:
                    ibin += jbin[self.q2]
                elif q == self.q2:
                    ibin += jbin[self.q1]
                else:
                    ibin += jbin[q]
            yield self.__bindec(ibin), 1

    def row(self, i0):
        """
    Devuelve un iterador sobre la fila i0 de la matriz de swap.
    :param i0: fila deseada.
    :return: iterador de la fila.
    """
        QuantumMatrix.row(self, i0)
        ibin = self.__binenc(i0)
        if ibin[self.c] == '0':
            yield i0, 1
        else:
            jbin = ''
            for q in range(self.nq):
                if q == self.q1:
                    jbin += ibin[self.q2]
                elif q == self.q2:
                    jbin += ibin[self.q1]
                else:
                    jbin += ibin[q]
            yield self.__bindec(jbin), 1

    def dot(self, M):
        """
    Calcula el producto punto de la matriz de swap y otra matriz.
    :param M: matriz a multiplicar.
    :return: matriz resultante del producto punto.
    """
        QuantumMatrix.dot(self, M)
        if type(M) == rho:
            new_rho = rho([], [], [], self.shape)
            for k, j in M.store.keys():
                for i, v in self.col(k):
                    if new_rho.get(i, j) == 0:
                        new_rho.set(i, j, v * M.get(k, j))
                    else:
                        new_rho.set(i, j, new_rho.get(i, j) + v * M.get(k, j))
            return new_rho
        else:
            return Dot(self, M)


class CSwap_reg(QuantumMatrix):
    """es una subclase de la clase QuantumMatrix y representa una matriz de compuerta de swap controlada por un qubit.
    La compuerta de swap controlada por un qubit actúa como la compuerta de swap regular solo si el estado del qubit de control es 1.
     Si el estado del qubit de control es 0, la compuerta no tiene efecto y la matriz es una matriz de identidad.
     La clase toma como entrada el índice del qubit de control, y dos listas de índices de qubits que se intercambiarán, así como el número total de qubits en el sistema
     """

    def __init__(self, c, qq1, qq2, nq):
        """
        Inicializa la matriz de intercambio controlado con las qubits de control, objetivo y el número de qubits totales.
        :param c: índice del qubit de control.
        :param qq1: índices de los primeros qubits a intercambiar.
        :param qq2: índices de los segundos qubits a intercambiar.
        :param nq: número total de qubits.
        """
        self.nq = nq
        self.c = c
        self.qq1 = qq1
        self.qq2 = qq2
        self.shape = (2 ** nq, 2 ** nq)
        self.nonzeros = 2 ** nq

    def __binenc(self, i):
        """
        Codifica el índice de la fila o columna en binario.
        :param i: índice a codificar.
        :return: cadena binaria que representa el índice.
        """
        ib = bin(i)[2:]
        if len(ib) < self.nq:
            ib = '0' * (self.nq - len(ib)) + ib
        return ib

    def __bindec(self, ib):
        """
        Decodifica una cadena binaria para obtener el índice correspondiente.
        :param ib: cadena binaria a decodificar.
        :return: índice correspondiente.
        """
        return sum(2 ** (len(ib) - j - 1) * int(ib[j]) for j in range(len(ib)))

    def get(self, i, j):
        """
        Devuelve el valor en la posición (i, j) de la matriz.
        :param i: fila.
        :param j: columna.
        :return: valor en la posición (i, j).
        """
        QuantumMatrix.get(self, i, j)
        ibin = self.__binenc(i)
        jbin = self.__binenc(j)
        if ibin[self.c] != jbin[self.c]:
            return 0
        if ibin[self.c] == '0':
            if i == j:
                return 1
            else:
                return 0
        else:
            for q in range(self.nq):
                if (q in self.qq1) or (q in self.qq2):
                    continue
                if ibin[q] != jbin[q]:
                    return 0
            if all(ibin[q1] == jbin[q2] and ibin[q2] == jbin[q1] for q1, q2 in zip(self.qq1, self.qq2)):
                return 1
            else:
                return 0

    def col(self, j0):
        """
        Devuelve un iterador sobre los elementos no nulos de una columna dada.
        :param j0: índice de la columna.
        :return: iterador sobre los elementos no nulos.
        """
        QuantumMatrix.col(self, j0)
        jbin = self.__binenc(j0)
        if jbin[self.c] == '0':
            yield j0, 1
        else:
            ibin = [b for b in jbin]
            for q1, q2 in zip(self.qq1, self.qq2):
                temp = ibin[q1]
                ibin[q1] = ibin[q2]
                ibin[q2] = temp
            yield self.__bindec(''.join(ibin)), 1

    def row(self, i0):
        """
        Devuelve un iterador sobre los elementos no nulos de la fila de la matriz correspondiente al índice dado.
        :param i0: índice de la fila.
        :return: iterador de tuplas (j, v) con los índices de columna y valores no nulos.
        """
        QuantumMatrix.row(self, i0)
        ibin = self.__binenc(i0)
        if ibin[self.c] == '0':
            yield i0, 1
        else:
            jbin = [b for b in ibin]
            for q1, q2 in zip(self.qq1, self.qq2):
                temp = jbin[q1]
                jbin[q1] = jbin[q2]
                jbin[q2] = temp
            yield self.__bindec(''.join(jbin)), 1

    def dot(self, M):
        """
        Calcula el producto punto de la matriz cuadrada y otra matriz.
        :param M: matriz a multiplicar.
        :return: matriz resultante del producto punto.
        """
        QuantumMatrix.dot(self, M)
        if type(M) == rho:
            new_rho = rho([], [], [], self.shape)
            for k, j in M.store.keys():
                for i, v in self.col(k):
                    if new_rho.get(i, j) == 0:
                        new_rho.set(i, j, v * M.get(k, j))
                    else:
                        new_rho.set(i, j, new_rho.get(i, j) + v * M.get(k, j))
            return new_rho
        else:
            return Dot(self, M)

class Identity(QuantumMatrix):

    def __init__(self, n):
        """
        Crea una matriz de identidad cuadrada de dimensiones nxn.
        :param n: dimensión de la matriz cuadrada.
        """
        self.n = n
        self.shape = (n, n)
        self.nonzeros = n

    def get(self, i, j):
        """
        Obtiene el valor almacenado en la posición (i,j) de la matriz.
        :param i: fila.
        :param j: columna.
        :return: valor en la posición (i,j).
        """
        QuantumMatrix.get(self, i, j)
        if i == j:
            return 1
        else:
            return 0

    def row(self, i0):
        """
        Devuelve un iterador sobre los elementos de la fila i0 de la matriz.
        :param i0: fila de la matriz.
        :return: iterador sobre los elementos de la fila i0.
        """
        QuantumMatrix.row(self, i0)
        yield i0, 1

    def col(self, j0):
        """
        Devuelve un iterador sobre los elementos de la columna j0 de la matriz.
        :param j0: columna de la matriz.
        :return: iterador sobre los elementos de la columna j0.
        """
        QuantumMatrix.col(self, j0)
        yield j0, 1

    def dot(self, M):
        """
        Calcula el producto punto de la matriz cuadrada y otra matriz.
        :param M: matriz a multiplicar.
        :return: matriz resultante del producto punto.
        """
        QuantumMatrix.dot(self, M)
        return M

class Oracle(QuantumMatrix):
    """
    Clase que representa un oráculo cuántico.

    Un oráculo es una matriz cuadrada que define una transformación en un sistema cuántico.
    Esta clase hereda de QuantumMatrix.

    Args:
        criteria (function): Función de criterio que define la acción del oráculo.
        nq (int): Número de qubits del sistema cuántico.
        reg1 (list): Lista de índices que representan los qubits del primer registro.
        reg2 (list): Lista de índices que representan los qubits del segundo registro.
        ancilla (int): Índice del qubit de la ancilla.

    Attributes:
        criteria (function): Función de criterio que define la acción del oráculo.
        nq (int): Número de qubits del sistema cuántico.
        reg1 (list): Lista de índices que representan los qubits del primer registro.
        reg2 (list): Lista de índices que representan los qubits del segundo registro.
        ancilla (int): Índice del qubit de la ancilla.
        shape (tuple): Tupla que indica las dimensiones de la matriz.
        nonzeros (int): Número de elementos no nulos en la matriz.

    Methods:
        get(i, j): Devuelve el valor del elemento (i, j) en la matriz.
        row(i0): Genera un iterador sobre los elementos de la fila i0.
        col(j0): Genera un iterador sobre los elementos de la columna j0.
        dot(M): Calcula el producto de la matriz con otra matriz o rho.
        get_qf_sort_oracle:  se utiliza para construir un oráculo de ordenamiento para un par de registros considerando una función de aptitud cuántica.


    """

    def __init__(self, criteria, nq, reg1, reg2, ancilla):
        """
        Inicializa un objeto de tipo Oracle.

        :param criteria: función booleana que representa el criterio del oráculo.
        :param nq: número de qubits en el registro.
        :param reg1: lista de índices de los qubits en el primer registro.
        :param reg2: lista de índices de los qubits en el segundo registro.
        :param ancilla: índice del qubit que actúa como ancilla en el circuito del oráculo.
        """
        if ancilla in reg1 or ancilla in reg2:
            raise ValueError('Ancilla qubit has to be out of reg1 and reg2')
        if any(q1 in reg2 for q1 in reg1):
            raise ValueError('reg1 and reg2 can not overlap')
        self.criteria = criteria
        self.nq = nq
        self.reg1 = reg1
        self.reg2 = reg2
        self.ancilla = ancilla

        self.shape = (2**nq, 2**nq)
        self.nonzeros = 2**nq

    def __binenc(self, i):
        """
        Codifica el número entero i en una cadena binaria de longitud nq.
        :param i: número entero a codificar.
        :return: cadena binaria que representa a i.
        """
        ib = bin(i)[2:]
        if len(ib) < self.nq:
            ib = '0' * (self.nq - len(ib)) + ib
        return ib

    def __bindec(self, ib):
        """
        Decodifica la cadena binaria ib en un número entero.
        :param ib: cadena binaria a decodificar.
        :return: número entero correspondiente a ib.
        """
        return sum(2 ** (len(ib) - j - 1) * int(ib[j]) for j in range(len(ib)))

    def get(self, i, j):
        """
        Devuelve un iterador sobre los elementos de la fila i0 de la matriz.
        :param i0: fila de la matriz a devolver.
        :return: iterador sobre los elementos de la fila i0.
        """
        ib = self.__binenc(i)
        jb = self.__binenc(j)

        for i, bibj in enumerate(zip(ib, jb)):
            bi, bj = bibj
            if bi != bj and i != self.ancilla:
                return 0

        qq1 = ''.join(ib[q] for q in self.reg1)
        qq2 = ''.join(ib[q] for q in self.reg2)

        a1 = jb[self.ancilla]
        a2 = ib[self.ancilla]

        doswap = self.criteria(qq1, qq2)

        if doswap:
            if a1 != a2:
                return 1
            else:
                return 0
        else:
            if a1 != a2:
                return 0
            else:
                return 1

    def row(self, i0):
        """
        Devuelve un iterador sobre los elementos de la fila i0 de la matriz.
        :param i0: fila de la matriz a devolver.
        :return: iterador sobre los elementos de la fila i0.
        """
        ib = self.__binenc(i0)
        qq1 = ''.join(ib[q] for q in self.reg1)
        qq2 = ''.join(ib[q] for q in self.reg2)
        a = ib[self.ancilla]
        doswap = self.criteria(qq1, qq2)

        if doswap:
            if a == '0':
                j = self.__bindec(''.join(bj if c != self.ancilla else '1' for c, bj in enumerate(ib)))
            else:
                j = self.__bindec(''.join(bj if c != self.ancilla else '0' for c, bj in enumerate(ib)))
        else:
            j = i0
        yield j, 1

    def col(self, j0):
        """
        Devuelve un iterador sobre los elementos de la columna j0 de la matriz.
        :param j0: columna de la matriz a devolver.
        :return: iterador sobre los elementos de la columna j0.
        """
        jb = self.__binenc(j0)
        qq1 = ''.join(jb[q] for q in self.reg1)
        qq2 = ''.join(jb[q] for q in self.reg2)
        a = jb[self.ancilla]
        doswap = self.criteria(qq1, qq2)

        if doswap:
            if a == '0':
                i = self.__bindec(''.join(bi if c != self.ancilla else '1' for c, bi in enumerate(jb)))
            else:
                i = self.__bindec(''.join(bi if c != self.ancilla else '0' for c, bi in enumerate(jb)))
        else:
            i = j0
        yield i, 1

    def dot(self, M):
        """
        Realiza el producto escalar entre la matriz del oráculo y otra matriz M.

        :param M: matriz a multiplicar.
        :return: matriz resultante del producto escalar.
        """
        QuantumMatrix.dot(self, M)
        if type(M) == rho:
            new_rho = rho([], [], [], self.shape)
            for k, j in M.store.keys():
                for i, v in self.col(k):
                    if new_rho.get(i, j) == 0:
                        new_rho.set(i, j, v * M.get(k, j))
                    else:
                        new_rho.set(i, j, new_rho.get(i, j) + v * M.get(k, j))
            return new_rho
        else:
            return Dot(self, M)

    @staticmethod
    def get_qf_sort_oracle(nq=5, reg1=np.array([0, 1]), reg2=np.array([2, 3]), ancilla=4, uu=None, criteria=None):
        """
        Static method to construct a sort oracle for a register pair considering a quantum fitness.
        Assumes registers are consecutive.
        Parameters
        ----------
        nq: int
            Total number of registers
        reg1: np.ndarray
            Representation of the first register. (upper level)
        reg2: np.ndarray
            Representation of the second register. (lower level)
        ancilla: int
            Position of the ancilla used.
        uu: np.ndarray
            Matrix representation of the change of basis from the canonical basis to the problem basis.
            Default to None, uu = I.
        criteria: function
            Function representing the criteria for the oracle.
            Signature: criteria(x,y) -> cmp
                x: string, representing the binary representation of a state for the upper register
                y: string, representing the binary representation of a state for the lower register
                cmp: boolean, True if f(x) < f(y); False if f(x) >= f(y). Where f is the fitness in the transformed basis.
        Returns
        -------
        oracle: np.ndarray
            Matrix representing the oracle in the canonical basis
        """
        if not all(i in np.concatenate((reg1, reg2)) for i in range(reg1[0], reg2[-1] + 1)):
            raise Exception("Only consecutive registers allowed for comparative.")

        if uu is None:
            uu = np.identity(2**(len(reg1)))

        if not isinstance(reg1, np.ndarray):
            reg1 = np.array(reg1)
        if not isinstance(reg2, np.ndarray):
            reg2 = np.array(reg2)

        if reg1.dtype != reg2.dtype:
            raise Exception("reg1 and reg2 must have the same data type.")

        uu = np.kron(np.kron(np.identity(2 ** reg1[0]), np.kron(uu, uu)),
                     np.identity(2 ** (nq - reg2[-1] - 1)))
        if criteria is None:
            criteria = lambda x, y: sum(int(xi) * 2 ** (len(x) - i - 1) for i, xi in enumerate(x)) > sum(
                int(yi) * 2 ** (len(y) - i - 1) for i, yi in enumerate(y))

        aux_oracle = Oracle(criteria, nq, reg1, reg2, ancilla)

        oracle = uu.dot(aux_oracle.get_matrix()).dot(np.transpose(uu.conjugate()))
        return oracle


# -------------------
#
#   Action matrices
#
# -------------------


class Transpose(QuantumMatrix):
    """
    Esta clase representa la matriz traspuesta.
    """

    def __init__(self, M):
        """
        Constructor de la clase Transpose.

        :param M: QuantumMatrix
            La matriz que se desea traspasar.
        """
        self.M = M
        self.shape = M.shape
        self.nonzeros = M.nonzeros

    def get(self, i, j):
        """
        Retorna el elemento en la posición (j, i) de la matriz traspuesta.

        :param i: int
            Índice de fila.
        :param j: int
            Índice de columna.
        :return: float
            Valor en la posición (j, i) de la matriz traspuesta.
        """
        return self.M.get(j, i)

    def col(self, i):
        """
        Retorna una columna de la matriz traspuesta.

        :param i: int
            Índice de la columna que se desea obtener.
        :return: generator
            Generador que retorna tuplas (j, v) donde j es el índice de la fila y v es el valor en la posición (j, i).
        """
        return self.M.row(i)

    def row(self, j):
        """
        Retorna una fila de la matriz traspuesta.

        :param j: int
            Índice de la fila que se desea obtener.
        :return: generator
            Generador que retorna tuplas (i, v) donde i es el índice de la columna y v es el valor en la posición (j, i).
        """
        return self.M.col(j)

    def dot(self, M):
        """
        Retorna el producto punto entre la matriz traspuesta y otra matriz dada.

        :param M: QuantumMatrix
            La matriz que se desea multiplicar con la matriz traspuesta.
        :return: QuantumMatrix
            Matriz resultante de la multiplicación.
        """
        QuantumMatrix.dot(self, M)
        if type(M) == rho:
            new_rho = rho([], [], [], self.shape)
            for k, j in M.store.keys():
                for i, v in self.col(k):
                    if new_rho.get(i, j) == 0:
                        new_rho.set(i, j, v * M.get(k, j))
                    else:
                        new_rho.set(i, j, new_rho.get(i, j) + v * M.get(k, j))
            return new_rho
        else:
            return Dot(self, M)

class HermitianConjugate(QuantumMatrix):
    """
    Clase que representa la conjugada hermitiana de una matriz cuántica.
    Hereda de la clase QuantumMatrix.
    """

    def __init__(self, M):
        """
        Inicializa la instancia de la clase HermitianConjugate.
        
        :param M: La matriz cuántica original.
        """
        self.M = M
        self.shape = M.shape
        self.nonzeros = M.nonzeros

    def get(self, i, j):
        """
        Devuelve el elemento (i, j) de la matriz como el conjugado del elemento (j, i) de la matriz original.
        
        :param i: Índice de fila.
        :param j: Índice de columna.
        :return: El conjugado del elemento (j, i) de la matriz original.
        """
        return self.M.get(j, i).conjugate()

    def col(self, i):
        """
        Itera a través de todos los elementos en la columna i de la matriz original y devuelve los elementos conjugados.
        
        :param i: Índice de columna.
        :yield: Tuplas (j, v) donde j es el índice de fila y v es el conjugado del elemento en la posición (j, i).
        """
        for j, v in self.M.row(i):
            yield j, v.conjugate()

    def row(self, j):
        """
        Itera a través de todos los elementos en la fila j de la matriz original y devuelve los elementos conjugados.
        
        :param j: Índice de fila.
        :yield: Tuplas (i, v) donde i es el índice de columna y v es el conjugado del elemento en la posición (j, i).
        """
        for i, v in self.M.col(j):
            yield i, v.conjugate()

    def dot(self, M):
        """
        Calcula el producto de la matriz Hermitiana conjugada y otra matriz cuántica.
        
        :param M: La matriz cuántica a multiplicar.
        :return: El resultado del producto de matrices, ya sea un objeto rho (si M es también un objeto rho) o un objeto Dot.
        """
        QuantumMatrix.dot(self, M)
        if type(M) == rho:
            new_rho = rho([], [], [], self.shape)
            for k, j in M.store.keys():
                for i, v in self.col(k):
                    if new_rho.get(i, j) == 0:
                        new_rho.set(i, j, v * M.get(k, j))
                    else:
                        new_rho.set(i, j, new_rho.get(i, j) + v * M.get(k, j))
            return new_rho
        else:
            return Dot(self, M)

class Dot(QuantumMatrix):
    """
    Clase que representa el producto de dos matrices cuánticas.
    Hereda de la clase QuantumMatrix.
    """

    def __init__(self, A, B):
        """
        Inicializa la instancia de la clase Dot.
        
        :param A: Primera matriz cuántica.
        :param B: Segunda matriz cuántica.
        """
        if A.shape != B.shape:
            raise IndexError(
                'Shape mismatch, obtained ({}, {}) and ({}, {})'.format(
                    A.shape[0], A.shape[1], B.shape[0], B.shape[1]))
        self.A = A
        self.B = B
        self.shape = A.shape
        self.nonzeros = A.shape[0] * A.shape[0] # This is a maximum

    def get(self, i, j):
        """
        Devuelve el elemento (i, j) de la matriz producto como la suma de los productos de los elementos correspondientes
        de las matrices A y B en las filas y columnas respectivas.
        
        :param i: Índice de fila.
        :param j: Índice de columna.
        :return: El elemento (i, j) de la matriz producto.
        """
        return sum(va * self.B.get(k, j) for k, va in self.A.row(i))

    def col(self, j0):
        """
        Itera a través de todos los elementos en la columna j0 de la matriz B y realiza el producto con los elementos
        correspondientes de la matriz A.
        
        :param j0: Índice de columna.
        :yield: Tuplas (i, v) donde i es el índice de fila y v es el producto de los elementos correspondientes.
        """
        QuantumMatrix.col(self, j0)
        ii = {}
        for k, vb in self.B.col(j0):
            for i, va in self.A.col(k):
                ii[i] = va*vb + ii.get(i, 0)
        return ii.items()


    def row(self, i0):
        """
        Itera a través de todos los elementos en la fila i0 de la matriz A y realiza el producto con los elementos
        correspondientes de la matriz B.
        
        :param i0: Índice de fila.
        :yield: Tuplas (j, v) donde j es el índice de columna y v es el producto de los elementos correspondientes.
        """
        QuantumMatrix.row(self, i0)
        jj = {}
        for k, va in self.A.row(i0):
            for j, vb in self.B.row(k):
                jj[j] = va*vb + jj.get(j, 0)
        return jj.items()

    def dot(self, M):
        """
        Calcula el producto de la matriz producto con otra matriz cuántica.
        
        :param M: La matriz cuántica a multiplicar.
        :return: El resultado del producto de matrices, ya sea un objeto rho (si M es también un objeto rho) o un objeto Dot.
        """
        QuantumMatrix.dot(self, M)
        if type(M) == rho:
            new_rho = rho([], [], [], self.shape)
            for k, j in M.store.keys():
                for i, v in self.col(k):
                    if new_rho.get(i, j) == 0:
                        new_rho.set(i, j, v * M.get(k, j))
                    else:
                        new_rho.set(i, j, new_rho.get(i, j) + v * M.get(k, j))
            return new_rho
        else:
            return Dot(self, M)


class Kronecker(QuantumMatrix):
    """
    Clase que representa la operación de Kronecker entre dos matrices cuánticas.
    Hereda de la clase QuantumMatrix.
    """

    def __init__(self, A, B):
        """
        Inicializa la instancia de la clase Kronecker.
        
        :param A: Primera matriz cuántica.
        :param B: Segunda matriz cuántica.
        """
        self.A = A
        self.na = A.shape[0]
        self.B = B
        self.nb = B.shape[0]
        self.shape = (self.na*self.nb, self.na*self.nb)
        self.nonzeros = A.nonzeros * B.nonzeros

    def get(self, i, j):
        """
        Devuelve el elemento (i, j) de la matriz resultante de la operación de Kronecker, que es el producto de los
        elementos correspondientes de las matrices A y B.
        
        :param i: Índice de fila.
        :param j: Índice de columna.
        :return: El elemento (i, j) de la matriz resultante.
        """
        return self.A.get(i//self.nb, j//self.nb) * self.B.get(i%self.nb, j%self.nb)

    def col(self, j0):
        """
        Itera a través de todos los elementos en la columna j0 de la matriz resultante y realiza el producto con los
        elementos correspondientes de las columnas j0//nb de la matriz A y j0%nb de la matriz B.
        
        :param j0: Índice de columna.
        :yield: Tuplas (i, v) donde i es el índice de fila y v es el producto de los elementos correspondientes.
        """
        QuantumMatrix.col(self, j0)
        ja = j0 // self.nb
        jb = j0 % self.nb

        for ia, va in self.A.col(ja):
            for ib, vb in self.B.col(jb):
                yield ia * self.nb + ib, va * vb

    def row(self, i0):
        """
        Itera a través de todos los elementos en la fila i0 de la matriz resultante y realiza el producto con los
        elementos correspondientes de las filas i0//nb de la matriz A y i0%nb de la matriz B.
        
        :param i0: Índice de fila.
        :yield: Tuplas (j, v) donde j es el índice de columna y v es el producto de los elementos correspondientes.
        """
        QuantumMatrix.row(self, i0)
        ia = i0 // self.nb
        ib = i0 % self.nb

        for ja, va in self.A.row(ia):
            for jb, vb in self.B.row(ib):
                yield ja*self.nb + jb, va*vb

    def dot(self, M):
        """
        Calcula el producto de la matriz resultante de la operación de Kronecker con otra matriz cuántica.
        
        :param M: La matriz cuántica a multiplicar.
        :return: El resultado del producto de matrices, ya sea un objeto rho (si M es también un objeto rho) o un objeto Dot.
        """
        QuantumMatrix.dot(self, M)
        if type(M) == rho:
            new_rho = rho([], [], [], self.shape)
            for k, j in M.store.keys():
                for i, v in self.col(k):
                    if new_rho.get(i, j) == 0:
                        new_rho.set(i, j, v * M.get(k, j))
                    else:
                        new_rho.set(i, j, new_rho.get(i, j) + v * M.get(k, j))
            return new_rho
        else:
            return Dot(self, M)

class KronExpand(QuantumMatrix):

    def __init__(self, n1, U, n2):
        """
        Inicializa la instancia de la clase KronExpand.
        
        :param n1: Número de veces que se repite U al principio de la matriz resultante.
        :param U: La matriz cuántica a expandir.
        :param n2: Número de veces que se repite U al final de la matriz resultante.
        """
        self.u = U
        self.n = U.shape[0]
        self.n1 = n1
        self.n2 = n2
        self.shape = (self.n1 * self.n * self.n2, self.n1 * self.n * self.n2)
        self.nonzeros = self.u.nonzeros * self.n1 * self.n2

    def get(self, i, j):
        """
        Devuelve el elemento (i, j) de la matriz expandida utilizando la propiedad Kronecker.
        
        :param i: Índice de fila.
        :param j: Índice de columna.
        :return: El elemento (i, j) de la matriz expandida.
        """
        if i//(self.n*self.n2) == j//(self.n*self.n2):
            if i%(self.n*self.n2)%self.n2 == j%(self.n*self.n2)%self.n2:
                return self.u.get((i%(self.n*self.n2))//self.n2, (j%(self.n*self.n2))//self.n2)
            else:
                return 0
        else:
            return 0

    def col(self, j0):
        """
        Itera a través de todos los elementos en la columna j0 de la matriz expandida y los obtiene a partir de los 
        elementos de la matriz U.
        
        :param j0: Índice de columna.
        :yield: Tuplas (i, v) donde i es el índice de fila y v es el valor del elemento correspondiente.
        """
        QuantumMatrix.col(self, j0)
        nu = j0 // (self.n*self.n2)
        pr = (j0 % (self.n*self.n2)) % self.n2
        for i, v in self.u.col((j0%(self.n*self.n2))//self.n2):
            yield i*self.n2 + pr + nu*self.n*self.n2, v

    def row(self, i0):
        """
        Itera a través de todos los elementos en la fila i0 de la matriz expandida y los obtiene a partir de los 
        elementos de la matriz U.
        
        :param i0: Índice de fila.
        :yield: Tuplas (j, v) donde j es el índice de columna y v es el valor del elemento correspondiente.
        """
        QuantumMatrix.row(self, i0)
        nu = i0 // (self.n * self.n2)
        pr = (i0 % (self.n * self.n2)) % self.n2
        for j, v in self.u.row((i0%(self.n*self.n2))//self.n2):
            yield j * self.n2 + pr + nu * self.n * self.n2, v

    def dot(self, M):
        """
        Realiza el producto entre la matriz expandida y otra matriz cuántica.
        :param M: La matriz cuántica a multiplicar.
        :return: El resultado del producto de matrices, ya sea un objeto rho (si M es también un objeto rho) o un objeto Dot.
        """
        QuantumMatrix.dot(self, M)
        if type(M) == rho:
            new_rho = rho([], [], [], self.shape)
            for k, j in M.store.keys():
                for i, v in self.col(k):
                    if new_rho.get(i, j) == 0:
                        new_rho.set(i, j, v * M.get(k, j))
                    else:
                        new_rho.set(i, j, new_rho.get(i, j) + v * M.get(k, j))
            return new_rho
        else:
            return Dot(self, M)


def test_cloning_routine(N=8, cl=4):
    clone = Identity(2 ** (N * cl))
    for nu in range(0, N // 2):
        s = Swap_reg(range((nu + 1) * cl, (nu + 2) * cl),
                     range((nu + N // 2) * cl, (nu + 1 + N // 2) * cl),
                     N * cl)
        u = Uclone(2 ** cl)
        unu = s.dot(KronExpand(2 ** (nu * cl), u, 2 ** ((N - nu - 2) * cl))).dot(s)
        clone = clone.dot(unu)

    inputs = [int(i * 2 ** (cl * N / 2)) for i in range(int(2 ** (cl * (N / 2 - 1))))]
    outputs = [int(i * 2 ** (cl * N / 2) + i) for i in range(int(2 ** (cl * (N / 2 - 1))))]

    for inp, ou in zip(inputs, outputs):
        r = rho([1], [inp], [inp], (2 ** (N * cl), 2 ** (N * cl)))

        # Here Transpose <-> HermitianConjugate
        r2 = clone.dot(r).dot(Transpose(clone))
        if r2.get(ou, ou) != 1:
            print('ALERT at ', inp, ou)


if __name__ == '__main__':
    theta = pi / 4
    H = rho([cos(theta), -cos(theta), sin(theta), sin(theta)], [0, 1, 0, 1], [0, 1, 1, 0],
               (2, 2))
    X = rho([1, 1], [0, 1], [1, 0], (2, 2))
    for i in range(4):
        r = rho.gen_rho_from_state([1/sqrt(2), 1/sqrt(2)], [0, 1], 2**2)
        print(r.str_qubit_state())
        r.projection_controlled_rotation(0, 1, X, H)
        print('->', r.str_qubit_state())
        print(r.trace())
        print()
    import numpy as np
    r2 = rho.gen_rho_from_matrix(np.array([[1, 1], [1, 1]]))
    print(r2)
