from typing import List, Tuple
import random
import pandas as pd
import numpy as np
import math

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
        by a random quantity in the range [-1,1] in both axis.
        """
        # Apply update and index on z_map.
        n_x, n_y = 0, 0
        while True:
            # Compute random movements until we get one that moves the agent within
            # the map boundaries (hopefully the first one)
            move_x, move_y = random.randint(-1,1)*random.random()*5, \
                            random.randint(-1,1)*random.random()*5
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

    - c is a parameter for the length of the sides of the simplex
    - alpha controls the placement of the new point in the "reflection"
        operation
    - beta controls the placement of the new point in the "contraction"
        operation
    - gamma controls the placement of the new point in the "expansion"
        operation
    - rho is the shrinking factor and controls the placement of all
        points when the shrink operation is chosen
    - epsilon is a threshold on the size of the area of the simplex
        when we check for convergence.
    '''
    def __init__(self, z_map: pd.DataFrame, c:float=10,
                alpha:float=1, gamma:float=1, beta:float=0.5, 
                rho:float=0.5, epsilon:float=0.01) -> None:
        super().__init__(z_map)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rho = rho
        self.epsilon = epsilon
        self.init_simplex(c)
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


class LineSearchOptimizer(Optimizer):
    ''' 
    Creates a simple Line Seach Optimizer which moves in the
    direction of a heuristically found direction with a step size 
    computed according to the Armijo condition.

    The heuristic for choosing the direction is:
    - Compute the height at all 8 adjacent points with respect to the
        one where the agent is currently at
    - Choose the direction of maximum (positive) change
    '''
    def next_step(self) -> Tuple:
        # To know the direction of movement we should know the gradient
        # or use a reasonable heuristic. In this case we don't know the
        # underlying function, so we have to use a heuristic.

        # 1. Compute the difference in height between the 
        #    current point and adjacent points
        #    Note: a positive difference means that there is a ascent, 
        #    a negative difference means that there is a descent
        current_z = self.get_z_level(self.x, self.y)
        #    Check boundaries
        xp = self.x+1 if self.x+1 < self.z_map_shape[0] else self.x
        xm = self.x-1 if self.x-1 >= 0 else self.x
        yp = self.y+1 if self.y+1 < self.z_map_shape[1] else self.y
        ym = self.y-1 if self.y-1 >= 0 else self.y
        adjacent_z = np.array([
            [ self.get_z_level(xm, ym), self.get_z_level(self.x, ym), self.get_z_level(xp, ym) ],
            [ self.get_z_level(xm, self.y), self.get_z_level(self.x, self.y), self.get_z_level(xp, self.y) ],
            [ self.get_z_level(xm, yp), self.get_z_level(self.x, yp), self.get_z_level(xp, yp) ]
        ])
        adjacent_diffs = adjacent_z - current_z

        # 2. Use the direction of max positive difference to move
        max_dir_idx = np.argmax(adjacent_diffs)
        #   // or % 3 because there are 3 neighbours per side,
        #   -1 because we have to change range [0,2] into range [-1,1]
        dir_x, dir_y = (max_dir_idx % 3) - 1, (max_dir_idx // 3) - 1

        # 3. Get an appropriate step length using Armijo conditions.
        # Note: We use as gradient approximation the depth difference in the two axis 
        # towards the movement direction
        gradient_approx = np.array([self.get_z_level(self.x + dir_x, self.y) - current_z,
                                    self.get_z_level(self.x, self.y + dir_y) - current_z])
        a = self.armijo_step_length(gradient_approx)

        # Once we have a direction, we need to scale it properly (being sure to stay within boundaries)
        n_x = self.x + a*gradient_approx[0] if 0 <= self.x + a*gradient_approx[0] < self.z_map_shape[0] else self.x
        n_y = self.y + a*gradient_approx[1] if 0 <= self.y + a*gradient_approx[1] < self.z_map_shape[1] else self.y
        n_z = self.get_z_level(n_x, n_y)
        
        # Update position on optimizer
        self.x = n_x
        self.y = n_y 
        return (n_x, n_y, n_z)


    def armijo_step_length(self, pk, c1=0.5, p=0.8, a_t=5):
        '''
        Algorithm to find a good step length using Armijo's condition. This condition
        ensures that the increase/decrease in the objective function should be proportional
        to both the step length and the gradient (or our approximation of it).

        Mathematically (for minimization), it's:

        f(xk+a*pk) < f(xk) + c1*a*np.dot(grad(xk).T@, pk)

        where xk is the current position, f is the map and the gradient
        is the height difference towards the direction of maximal change in the 
        two axis (approximation of the gradient)

        Input:
        -    pk: current direction (the gradient approximation) ([d_x, d_y])
        -    grad: gradient function of f (or an approximation)
        -    a_t: starting a (>0)
        -    p: scalar of direction for finding new a, [0,1]
        -    c1: scalar that has an impact on the difference that must be present between xk and xk+1
        
        Returns:
        -    a: suggested step size for this iteration
        '''
        xk = np.array([self.x, self.y])
        a = a_t
        j = 0
        while j < 10:
            new_pos = xk+a*pk
            if  self.get_z_level(new_pos[0], new_pos[1]) > \
                self.get_z_level(xk[0], xk[1]) + c1*a*np.dot(pk.T,pk): # Armijo condition
                break
            else:
                a = p*a
                j += 1
        return a