from typing import List, Tuple
import random
import pandas as pd
import numpy as np
import math
from constants import TUNING_SIZE, DESIRED_SIZE

class Optimizer():
    '''
    Base class for all optimizers.
    '''
    def __init__(self, z_map:pd.DataFrame) -> None:
        self.z_map = z_map
        self.z_arr = np.array(self.z_map)
        self.z_map_shape = self.z_map.shape
        start_x, start_y, _ = self.get_starting_point()
        self.x = start_x
        self.y = start_y
        self.scale = DESIRED_SIZE / TUNING_SIZE # Used for map size scaling

    def in_boundaries(self, x:float, y:float) -> bool:
        """
        Checks if a x,y position is within the boundaries of the map
        """
        return 0 <= x <= self.z_map_shape[0] - 1 and 0 <= y <= self.z_map_shape[1] - 1

    def get_z_level(self, x:float, y:float) -> float:
        '''
        Returns the z level at any arbitrary floating point coordinate within boundaries
        using bilinear interpolation for approximating its value.
        If the point is out of boundaries, its height level is automatically set to -1.
        '''
        # Note that on z_map dataframe y refers to the row and x to the column.
        if not self.in_boundaries(x, y):
            # If the point is not in boundaries, its height is -1.
            return -1
        if int(x) == x and int(y) == y:
            # Shortcut if both coordinates are int
            return self.z_map.iloc[int(y),int(x)]
        # Compute coordinates and height of surrounding exact positions
        lx = math.floor(x)
        ux = math.ceil(x) if math.ceil(x) != lx else lx+1 # For ints
        ly = math.floor(y)
        uy = math.ceil(y) if math.ceil(y) != ly else ly+1 # For ints
        # Check if upper bounds are within boundaries
        if not self.in_boundaries(ux, uy):
            # Height is -1 outside of boundaries: negative height will
            # never be chosen by direction algorithms
            z1, z2, z3, z4 = self.z_map.iloc[ly, lx], -1,-1,-1
        else:
            z1, z2, z3, z4 = self.z_map.iloc[ly, lx], self.z_map.iloc[ly, ux],\
                             self.z_map.iloc[uy, lx], self.z_map.iloc[uy, ux]
        # Interpolate between upper and lower points (weighted sum of influences)
        r1 = z1*(ux-x)/(ux-lx) + z2*(x-lx)/(ux-lx)
        r2 = z3*(ux-x)/(ux-lx) + z4*(x-lx)/(ux-lx)
        # Apply y-axis interpolation to get the final height
        z_final = r1*(uy-y)/(uy-ly)+r2*(y-ly)/(uy-ly)
        return z_final

    def get_starting_point(self) -> Tuple:
        '''
        Computes the starting point of the actor (a position in the low 
        area of the mountain, not necessarily the lowest because 
        most methods need a good initial guess to get to the
        optima and we try to deal with this need through randomness)
        ''' 
        row, col = random.choice(np.argwhere(
            self.z_arr < self.z_arr.min() + ((self.z_arr.max() - self.z_arr.min()) / 10))
        )
        # Old code, used to calculate the minimum point
        #col = np.argmin(self.z_arr) % self.z_map_shape[1]
        #row = math.floor(np.argmin(self.z_arr) / self.z_map_shape[1])
        return (col, row, self.z_arr[row,col])

    def next_step(self) -> Tuple:
        """
        Computes next step. In its base version it simply returns the 
        current position, but it must be overloaded to whatever strategy
        we choose to adopt.
        It must always return a tuple containing the new position of the 
        agent (x,y,z).
        """
        return (self.x, self.y, self.get_z_level(self.x, self.y))


class RandomOptimizer(Optimizer):
    '''
    Moves the agent randomly.
    '''
    def next_step(self) -> Tuple:
        """
        The following position of the agent is found by moving the agent
        by a random quantity in the range `[-5*scale, +5*scale]` in both axis.
        """
        # Apply update and index on z_map.
        n_x, n_y = 0, 0
        while True:
            # Compute random movements until we get one that moves the agent within
            # the map boundaries (hopefully the first one)
            move_x, move_y = random.randint(-1,1)*random.random()*self.scale, \
                            random.randint(-1,1)*random.random()*self.scale
            n_x, n_y =  (self.x+move_x), (self.y+move_y) 
            if self.in_boundaries(n_x, n_y):
                break
        n_z = self.get_z_level(n_x, n_y)
        # Update position on optimizer
        self.x = n_x
        self.y = n_y
        return (n_x, n_y, n_z)


class NelderMeadOptimizer(Optimizer):
    '''
    Creates an optimizer that uses the Nelder-Mead Simplex method
    to find an optima for the function. The method does not guarantee
    a global optima. It does not use derivatives, but simply applies 
    some transformations to an initial simplex (the generalization of a 
    triangle in an n dimensional space). The position of the agent is 
    determined as the center point of the simplex.

    Note: this optimizer assumes that the space of choice for positions
    is bidimentional (coordinates [x,y])
    ---
    - c is a parameter for the length of the sides of the simplex (default: 10*scale).
    - alpha controls the placement of the new point in the "reflection"
        operation (default: 1).
    - beta controls the placement of the new point in the "contraction"
        operation (default: 0.5).
    - gamma controls the placement of the new point in the "expansion"
        operation (default: 1).
    - rho is the shrinking factor and controls the placement of all
        points when the shrink operation is chosen (default: 0.5).
    - epsilon is a threshold on the size of the area of the simplex
        when we check for convergence (default: 0.01*scale).
    '''
    def __init__(self, z_map: pd.DataFrame, c:float=10,
                alpha:float=1, gamma:float=1, beta:float=0.5, 
                rho:float=0.5, epsilon:float=0.01) -> None:
        super().__init__(z_map)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rho = rho
        self.epsilon = epsilon*self.scale
        self.init_simplex(c*self.scale)
        self.rank_points()

    def init_simplex(self, c:float) -> None:
        '''
        Initializes the simplex. 

        Parameters:
        - c: the length of the sides of the initial simplex
        '''
        # Initial point of the simplex
        self.x0 = np.array([self.x, self.y])
        # The other points are at a distance c from this point.
        # Given b = (c/(n*math.sqrt(2)))*(math.sqrt(n+1)-1)
        # n being the number of dimensions, and
        # a = b + c/math.sqrt(2),
        # the other points are found by adding a vector of b to x0
        # except for the i-th element which is a.
        self.n = 2 # 2 dimensions --> simplex is a triangle
        b = c*(math.sqrt(self.n+1)-1)/(self.n*math.sqrt(2))
        a = b + c/math.sqrt(2)
        self.x1 = self.x0 + np.array([a,b])
        self.x2 = self.x0 + np.array([b,a])
        # Create a list keeping all the points of the current simplex
        self.xs = [ self.x0,self.x1,self.x2 ]

    def rank_points(self) -> None:
        '''
        Ranks the points of the simplex:
        - The one having the highest value on the objective function is the "best"
        - The one having the lowest value on the objective function is the "worst"
        - The one having the second lowest value on the objective function is the "lousy"
        '''
        # Sort the list of simplex points by their height
        sorted_points = sorted(self.xs, key=lambda x:self.get_z_level(x[0],x[1]))
        # Best = highest point
        self.xb = sorted_points[-1]
        # Worst = lowest point
        self.xw = sorted_points[0]
        # Lousy = second lowest point
        self.xl = sorted_points[1]
        # Also save the function values for those points
        self.fb, self.fw, self.fl = self.get_z_level(self.xb[0],self.xb[1]),\
                                    self.get_z_level(self.xw[0],self.xw[1]),\
                                    self.get_z_level(self.xl[0],self.xl[1])

    def reflection(self) -> np.array:
        '''
        Produces a point reflecting the simplex on one side.
        '''
        return self.xa + self.alpha*(self.xa - self.xw)

    def expansion(self) -> np.array:
        '''
        Produces a point by creating a larger simplex on the side of expansion
        '''
        return self.xr + self.gamma*(self.xr - self.xa)

    def contraction(self, inside=True) -> np.array:
        '''
        Contraction can be:
        - Inside contraction (reduce area of simplex by shrinking worst point)
        - Outside contraction (reduce area of simplex by expanding in opposite direction)
        '''
        return self.xa + self.beta*(-1 if inside else 1)*(self.xa-self.xw)

    def shrink(self) -> List[np.array]:
        '''
        Shrinking changes every point but the best one.
        Always ends perturbation.
        '''
        new_plist = [ self.xb + self.rho*(x-self.xb) 
            for x in self.xs if np.all(x != self.xb) ]
        new_plist.append(self.xb)
        self.xs = new_plist
        return new_plist

    def end_perturbation(self) -> None:
        self.xs = [ self.xb, self.xl, self.xw ]

    def converged(self) -> None:
        # The convergence criterion is based on the area of the simplex
        return np.sum(np.abs(np.array(self.xs[:-1])-self.xs[-1])) < self.epsilon

    def simplex_perturbation(self) -> Tuple:
        # Compute xa, the average of the points in the simplex excluding xw
        #   Remove the array matching with xw from xw
        points_no_xw = np.asarray(self.xs)[np.all(self.xs != self.xw, axis=1)]
        #   Compute the average of the remaining points
        self.xa = np.average(points_no_xw, axis=0)
        # REFLECT to get xr
        self.xr = self.reflection()
        #    Evaluate xr
        fr = self.get_z_level(self.xr[0], self.xr[1])
        # FIRST CHECK: is the reflected point better than the best?
        if fr > self.fb:
            # Try performing an EXPANSION to get further in that direction.
            # Obtain xe and evaluate it
            self.xe = self.expansion()
            fe = self.get_z_level(self.xe[0], self.xe[1])
            # Is the expanded point better than the best point?
            if fe > self.fb:
                # Accept the expansion: xw becomes xe
                self.xw = self.xe
                self.fw = fe
                # Update list of points and end epoch
                self.end_perturbation()
            else:
                # xe is not better than xb, but we already got xr better than xb, so 
                # we use that for the update
                self.xw = self.xr
                self.fw = fr
                self.end_perturbation()
        # SECOND CHECK: is the reflected point better than the lousy?
        elif fr >= self.fl:
            # It means it's also better than the worst, so we simply keep it
            self.xw = self.xr
            self.fw = fr
            self.end_perturbation()
        else:
            # If we reach this part, it means that the reflected point is either
            # worse than the worst or between the worse and the lousy (slightly better
            # than the worse)
            # THIRD CHECK: is the reflected point worse than the worst?
            if fr < self.fw:
                # Perform an INSIDE CONTRACTION and evaluate the new point
                self.xc = self.contraction(inside=True)
                fc = self.get_z_level(self.xc[0], self.xc[1])
                # If this point is better than the worse, keep it
                if fc > self.fw:
                    self.xw = self.xc
                    self.fw = fc
                    self.end_perturbation()
                else:
                    # The new point is still the worst: shrink the simplex
                    self.xs = self.shrink()
            else:
                # The reflected point is slightly better than the worst
                # Perform an OUTSIDE CONTRACTION and evaluate them
                self.xc = self.contraction(inside=False)
                fc = self.get_z_level(self.xc[0], self.xc[1])
                # LAST CHECK: is the contracted point better than the reflected point?
                if fc >= fr:
                    # If it is, keep it
                    self.xw = self.xc
                    self.fw = fc
                    self.end_perturbation()
                else:
                    # Shrink the simplex
                    self.xs = self.shrink()

    def get_position(self) -> Tuple:
        pos = list(np.average(np.array(self.xs), axis=0))
        pos.append(self.get_z_level(pos[0], pos[1]))
        return tuple(pos)

    def next_step(self) -> Tuple:
        if not self.converged():
            self.rank_points()
            self.simplex_perturbation()
        return self.get_position()


class BacktrackingLineSearchOptimizer(Optimizer):
    ''' 
    Creates a simple Backtracking Line Seach Optimizer which moves in a
    heuristically-computed direction with a dynamic step size.

    The direction is computed as the gradient of the function,
    estimated through numerical differentiation and the method of
    finite differences.
    - Rather than evaluating the derivatives at point x, we calculate
        the slope of a line that passes from x-h and x+h given h small number.
        Thus, the derivative is approximated to (f(x+h)-f(x-h)) / 2h (symmetric
        difference quotient)
    - The step size is found by starting with a very large step and iteratively 
        decreasing it until the update respects the Armijo condition (until the 
        increase in the objective function corresponds to an expected increase)
    ---
    - h: A parameter that indicates how much should we look around for calculating
        the derivative (default: 3*scale).
    - c1: Scalar that has an impact on the difference that must be present between xk and xk+1
        for a step size to be considered appropriate (default: 0.5).
    - p: Scalar to be used as a scaler to reduce the step size at the following iteration of
        the backtracking algorithm (default: 0.8).
    - a_t: Starting step size to be iteratively decreased until Armijo condition is respected 
        (default: 10*scale).
    '''
    def __init__(self, z_map: pd.DataFrame, h:float=2, c1:float=0.5, 
                 p:float=0.8, a_t:float=10) -> None:
        super().__init__(z_map)
        self.h = h*self.scale
        self.c1 = c1
        self.p = p
        self.a_t = a_t*self.scale

    def gradient_approx(self) -> np.array:
        '''
        Computes a gradient approximation ([dx, dy]) using finite differences.
        '''
        # Derivative in x axis: (f(x+h, y)-f(x-h, y)) / 2h
        dx = self.get_z_level(self.x+self.h, self.y) - self.get_z_level(self.x-self.h, self.y) / 2*self.h
        # Derivative in y axis: (f(x, y+h)-f(x, y-h) / 2h
        dy = self.get_z_level(self.x, self.y+self.h) - self.get_z_level(self.x, self.y-self.h) / 2*self.h
        # Gradient approximation: [dx,dy]
        return np.array([dx, dy])

    def next_step(self) -> Tuple:
        # Compute the gradient approximation
        grad = self.gradient_approx()
        # The "gradient" is the proposed direction of movement. Using Armijo condition we explore 
        # what could be a good step size choice.
        a = self.backtracking_algorithm(grad)

        # Once we have a direction, we need to scale it properly (being sure to stay within boundaries)
        n_x = self.x + a*grad[0] #if 0 <= self.x + a*grad[0] < self.z_map_shape[0] else self.x
        n_y = self.y + a*grad[1] #if 0 <= self.y + a*grad[1] < self.z_map_shape[1] else self.y
        n_z = self.get_z_level(n_x, n_y)
        
        # Update position on optimizer
        self.x = n_x
        self.y = n_y 
        return (n_x, n_y, n_z)


    def backtracking_algorithm(self, pk):
        '''
        Algorithm to find a good step length iteratively, starting from a high step size and
        iteratively decreasing it until the update respects Armijo's condition. This condition
        ensures that the increase/decrease in the objective function should be proportional
        to both the step length and the our gradient approximation.

        Mathematically (for minimization), Armijo condition is:

        f(xk+a*pk) < f(xk) + c1*a*np.dot(grad(xk).T@, pk)

        where xk is the current position, pk the proposed direction,
        grad(xk) the (approximation) of the gradient (in our case, pk
        and the gradient are the same thing).

        Note: the algorithm could techincally take a very long time to converge and
        this time could be better used to compute another direction, so we let it 
        run for at most 15 iterations (min step_size = about 0.35*scale).
        ---
        Input:
        -    pk: current direction (the gradient approximation) ([d_x, d_y])
        ---
        Returns:
        -    a: suggested step size for this iteration
        '''
        # Get the current position into an array
        xk = np.array([self.x, self.y])
        # Initialize the starting step size
        a = self.a_t
        j = 0
        while j < 15:
            # Compute the new position 
            new_pos = xk+a*pk
            # Armijo condition: if respected we have found a good step size
            if  self.get_z_level(new_pos[0], new_pos[1]) >= \
                self.get_z_level(xk[0], xk[1]) + self.c1*a*np.dot(pk.T,pk): 
                break
            else:
                # Otherwise, reduce the step size and retry
                a = self.p*a
                j += 1
        return a