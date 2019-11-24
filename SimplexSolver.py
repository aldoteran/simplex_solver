#!usr/bin/env python

"""
Class script for a Simplex solver for linear programming
applications

Author: Aldo Teran <aldot@kth.se>
"""

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import pdb

class SimplexSolver:

    def __init__(self, A_mat, b_vec, c_vec, is_min=True, verbose=True):
        """
        Initialize the object with the 'A' matrix,
        the vector 'b' with the contraints, vector
        'c' with the associated costs. It is assumed
        that the problem is already in standard form.
        """

        # Assign variables to class attributes
        self.A_mat = A_mat
        self.b_vec = b_vec
        self.c_vec = c_vec
        self.is_min = is_min
        self.iteration = 0
        self.verbose = verbose

    def plot_2D_problem(self):
        # Will plot the constraints of the problem
        self.fig = plt.figure()
        self.ax = plt.axes()

        [m, n] = self.A_mat.shape

        # Create a dictionary with variables
        variables = {}
        for i in range(n):
            variables["x_{0}".format(i)] = np.linspace(0, max(self.b_vec)*1.5, 10)

        # Plot constraints with x_2 on the vertical axis
        for i in range(m):
            x_2 = (self.b_vec[i] - variables['x_0'] * self.A_mat[i, 0]) / self.A_mat[i ,1]
            plt.plot(variables['x_0'], x_2, label=r'$f_{0}$'.format(i))

        plt.xlim((0, max(self.b_vec)))
        plt.ylim((0, max(self.b_vec)))
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.legend()

    def run_simplex(self, beta, nu, A_mat, b_vec, c_vec):
        """
        Runs one iteration of the simplex algorithm on
        the given problem.
        """
        if self.verbose:
            print("----------------------------------------------------")
            print("Starting iteration number {0}".format(self.iteration))

        # First off, get the dimensions of the problem
        [m, n] = A_mat.shape

        # Separate A_beta and A_nu
        A_beta = A_mat[:, beta]
        A_nu = A_mat[:, nu]

        # Separate costs
        c_beta = c_vec[beta]
        c_nu = c_vec[nu]

        # Calculate the value (b_bar) for our basic variables (A_beta * b_bar = b)
        A_beta_inv = inv(A_beta)
        b_bar = np.dot(A_beta_inv, b_vec)

        if self.verbose:
            print("A_beta:")
            print(A_beta)
            print("A_nu:")
            print(A_nu)
            print("c_beta:")
            print(c_beta)
            print("c_nu:")
            print(c_nu)
            print("b_bar:")
            print(b_bar)

        # Calculate simplex multipliers (y) with (A_beta^T * y = c_beta)
        y = np.dot(A_beta_inv.transpose(), c_beta)

        # Compute reduce cost for non basic variables (r_nu)
        # with (r_nu = c_nu - A_nu^T * y)
        r_nu = c_nu - np.dot(A_nu.transpose(), y)

        if self.verbose:
            print("Simplex multipliers y:")
            print(y)
            print("Reduced cost for iteration {0}".format(self.iteration))
            print(r_nu)

        # Done if elements in r_nu nonegative
        if min(r_nu) >= 0:
            print("Found optimal solution! The optimal x is:")
            x = np.vstack((b_bar, np.zeros((m,1))))
            print(x)
            print("Optimal value is:")
            print(np.dot(c_vec.transpose(), x))

            # Create an empty vector for x and fill it with b_bar values
            x = np.zeros(n)
            x[beta] = b_bar

            return [x, beta, nu, True]

        # Get min of r_nu to determine which index (q) enters the basis
        q = r_nu.argmin()

        # Determine vector (a_bar) with (A_beta * a_bar = a_q)
        a_bar = np.asarray(np.dot(A_beta_inv, A_mat[:, [q]]))

        if self.verbose:
            print("Index {0} enters the basis".format(nu[q]))
            print("a_bar:")
            print(a_bar)

        # Check if values are negative or 0
        if (a_bar <= 0).all():
            print("Solution does not exits! Sorry!")
            print("exiting...")
            return -1

        # Find the index that leaves the basis
        t_max = 9999999
        i = 0
        for val in a_bar:
            if val > 0:
                temp = b_bar[i] / val
                if temp < t_max:
                    t_max = temp
                    p = i
            i += 1

        if self.verbose:
            print("Index {0} leaves the basis".format(beta[p]))

        # Create an empty vector for x and fill it with b_bar values
        x = np.zeros(n)
        x[beta] = b_bar

        # We now know that index q replaces idx p in the basis
        new_basis = nu[q]
        nu[q] = beta[p]
        beta[p] = new_basis

        if self.verbose:
            print("Solution for iteration {0} is:".format(self.iteration))
            print(x)
            print("The new basis beta is:")
            print(beta)

        self.iteration += 1

        return [x, beta, nu, False]

    def find_solution(self, beta, nu, A_mat, b_vec, c_vec):
        """
        Runs the run_simplex method iteratively until
        a solution is found, given an initial solution.
        Returns the
        """
        finished = False
        while not finished:
            # The solver will return True when optimal solution is found
            [x, beta, nu, finished] = self.run_simplex(beta, nu, A_mat, b_vec, c_vec)

        return [x, beta, nu]

    def find_initial_bfs(self, A_mat, b_vec, c_vec):
        """
        Finds an intial Basic Feasible Solution using
        slack variables to solve an artificial linear
        program (P').
        """

        # Add slack vars to A
        [rows, cols] = np.shape(A_mat)
        A_mat = np.hstack((A_mat, np.eye(rows)))
        c_vec = np.vstack((np.zeros((cols, 1)), np.ones((rows, 1))))
        [rows, cols] = np.shape(A_mat)

        # Create index list for nu and beta values
        idx_list = [i for i in range(cols)]

        # Beta indices will correspond to the slack variables
        beta_idx = idx_list[-rows:]

        # Nu indices will be the leftovers
        nu_idx = idx_list[:-rows]

        if self.verbose:
            print("Attempting to find initial Basic Feasible Solution with slack variables")
            print("Artificial linear program P':")
            print("A:")
            print(A_mat)
            print("b:")
            print(b_vec)
            print("c:")
            print(c_vec)

            print("Starting basis with slack variables")
            print("beta:")
            print(beta_idx)
            print("nu:")
            print(nu_idx)

        # Run simplex with new matrices and initial basic feasible solution
        [x_opt, beta, nu] = self.find_solution(beta_idx, nu_idx, A_mat, b_vec, c_vec)
        # Case no. 1: optimal solution for P' is positive
        if x_opt.all() > 0:
            print("The linear program (P) has no basic feasible solution!")
            print("exiting program...")
            return -1
        # Case no. 2: optimal solution for P' is 0
        elif x_opt.all() == 0:
            print("Found optimal basic feasible solution for P', the solution is:")
            print(x_opt)

        print("Updating basic feasible solution for linear problem P...")

        self.iteration = 0

        # Return indices corresponding to the BFS of primal problem P
        return [beta, nu]

def main():
    """
    Main function to solve the linear programming example 5.4
    in the book using the SimplexSolver class above.
    """

    # Let's try solving example 5.4 from the book
    A = np.array([[1,1,1,0],
                  [2,1,0,1]])
    b = np.array([[200],
                  [300]])
    c = np.array([[-400],
                  [-300],
                  [0],
                  [0]])

    # We instantiate the simplex solver class with the arrays above
    simplex = SimplexSolver(A, b, c)
    simplex.plot_2D_problem()
    plt.show()

    # The last two variables are our slack variables so we start
    # with them as out initial basic variables.
    beta = [2,3] # Python starts indexing at 0
    nu = [0,1]

    # Now that we have eveything set up, we start running the solver
    finished = False
    while not finished:
        # The solver will return True when optimal solution is found
        [x, beta, nu, finished] = simplex.run_simplex(beta, nu, A, b, c)

    # x = np.expand_dims(x,1)
    print("Resulting x is:")
    print(x)
    print("Min cost is:")
    print(np.dot(c.transpose(), x))

if __name__ == "__main__":
    main()





