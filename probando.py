from abc import ABC, abstractmethod


class QuantumMatrix:
    error_tol = 1e-5
    @abstractmethod
    def get(self, i, j):
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
        if self.shape != M.shape:
            raise IndexError(
                'Shape mismatch, obtained ({}, {}) and ({}, {})'.format(
                    self.shape[0], self.shape[1], M.shape[0], M.shape[1]))
        pass

    def print_mat(self, block=None, len_lim=32):
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
        if self.shape[0] > len_lim:
            raise ValueError('Too big for getting matrix with len_lim={}. shape = {}'.format(len_lim, self.shape))
        mat = np.zeros(self.shape)
        for i in range(self.shape[0]):
            for j, v in self.row(i):
                mat[i, j] = v
        return mat
