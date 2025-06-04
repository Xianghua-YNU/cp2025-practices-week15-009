#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目2：打靶法与scipy.solve_bvp求解边值问题 - 学生代码模板

本项目要求实现打靶法和scipy.solve_bvp两种方法来求解二阶线性常微分方程边值问题：
u''(x) = -π(u(x)+1)/4
边界条件：u(0) = 1, u(1) = 1

学生姓名：[YOUR_NAME]
学号：[YOUR_STUDENT_ID]
完成日期：[COMPLETION_DATE]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp, solve_bvp
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')


def ode_system_shooting(t, y):
    """
    Define the ODE system for shooting method.
    
    Convert the second-order ODE u'' = -π(u+1)/4 into a first-order system:
    y1 = u, y2 = u'
    y1' = y2
    y2' = -π(y1+1)/4
    
    Args:
        t (float): Independent variable (time/position)
        y (array): State vector [y1, y2] where y1=u, y2=u'
    
    Returns:
        list: Derivatives [y1', y2']
    
    TODO: Implement the ODE system conversion
    Hint: Return [y[1], -np.pi*(y[0]+1)/4]
    """
    # TODO: Implement ODE system for shooting method
    y = np.asarray(y)
    return [y[1], -np.pi*(y[0]+1)/4]


def boundary_conditions_scipy(ya, yb):
    """
    Define boundary conditions for scipy.solve_bvp.
    
    Boundary conditions: u(0) = 1, u(1) = 1
    ya[0] should equal 1, yb[0] should equal 1
    
    Args:
        ya (array): Values at left boundary [u(0), u'(0)]
        yb (array): Values at right boundary [u(1), u'(1)]
    
    Returns:
        array: Boundary condition residuals
    
    TODO: Implement boundary conditions
    Hint: Return np.array([ya[0] - 1, yb[0] - 1])
    """
    # TODO: Implement boundary conditions for scipy.solve_bvp
    return np.array([ya[0] - 1, yb[0] - 1])

def ode_system_scipy(x, y):
    """
    Define the ODE system for scipy.solve_bvp.
    
    Note: scipy.solve_bvp uses (x, y) parameter order, different from odeint
    
    Args:
        x (float): Independent variable
        y (array): State vector [y1, y2]
    
    Returns:
        array: Derivatives as column vector
    
    TODO: Implement ODE system for scipy.solve_bvp
    Hint: Use np.vstack to return column vector
    """
    # TODO: Implement ODE system for scipy.solve_bvp
    return np.vstack((y[1], -np.pi*(y[0]+1)/4))

def solve_bvp_shooting_method(x_span, boundary_conditions, n_points=100, max_iterations=10, tolerance=1e-6):
    """
    Solve boundary value problem using shooting method.
    
    Algorithm:
    1. Guess initial slope m1
    2. Solve IVP with initial conditions [u(0), m1]
    3. Check if u(1) matches boundary condition
    4. If not, adjust slope using secant method and repeat
    
    Args:
        x_span (tuple): Domain (x_start, x_end)
        boundary_conditions (tuple): (u_left, u_right)
        n_points (int): Number of discretization points
        max_iterations (int): Maximum iterations for shooting
        tolerance (float): Convergence tolerance
    
    Returns:
        tuple: (x_array, y_array) solution arrays
    
    TODO: Implement shooting method algorithm
    Hint: Use secant method to adjust initial slope
    """
    # TODO: Validate input parameters
    
    # TODO: Extract boundary conditions and setup domain
    
    # TODO: Implement shooting method with secant method for slope adjustment
    
    # TODO: Return solution arrays

    # 确保初始条件为数组形式
    y0 = np.array([u_left, initial_slope_guess], dtype=float)

    # 使用solve_ivp求解（注意参数顺序）
    sol = solve_ivp(fun=ode_system_shooting, 
                    t_span=x_span,
                    y0=y0,
                    t_eval=np.linspace(x_start, x_end, n_points))
    
    # Validate input parameters
    if len(x_span) != 2 or x_span[1] <= x_span[0]:
        raise ValueError("x_span must be a tuple (x_start, x_end) with x_end > x_start")
    if len(boundary_conditions) != 2:
        raise ValueError("boundary_conditions must be a tuple (u_left, u_right)")
    if n_points < 10:
        raise ValueError("n_points must be at least 10")
    
    x_start, x_end = x_span
    u_left, u_right = boundary_conditions
    
    # Function to solve IVP and return error at right boundary
    def shooting_error(slope):
        sol = solve_ivp(ode_system_shooting, x_span, [u_left, slope], 
                        t_eval=np.linspace(x_start, x_end, n_points))
        return sol.y[0, -1] - u_right
    
    # Initial guesses for slope
    m_prev = 0.0
    m_curr = -1.0
    
    # Solve with initial guesses
    error_prev = shooting_error(m_prev)
    error_curr = shooting_error(m_curr)
    
    # Secant method iteration
    for iteration in range(max_iterations):
        if abs(error_curr) < tolerance:
            break
            
        # Avoid division by zero
        if abs(error_curr - error_prev) < 1e-12:
            m_next = m_curr + 0.1
        else:
            m_next = m_curr - error_curr * (m_curr - m_prev) / (error_curr - error_prev)
        
        # Update for next iteration
        m_prev, m_curr = m_curr, m_next
        error_prev, error_curr = error_curr, shooting_error(m_next)
    
    # Final solution with best slope
    final_slope = m_curr
    sol = solve_ivp(ode_system_shooting, x_span, [u_left, final_slope], 
                    t_eval=np.linspace(x_start, x_end, n_points))
    
    if abs(error_curr) > tolerance:
        print(f"Warning: Shooting method did not fully converge. Final error: {abs(error_curr):.2e}")
    
    return sol.t, sol.y[0]


def solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points=50):
    """
    Solve boundary value problem using scipy.solve_bvp.
    
    Args:
        x_span (tuple): Domain (x_start, x_end)
        boundary_conditions (tuple): (u_left, u_right)
        n_points (int): Number of initial mesh points
    
    Returns:
        tuple: (x_array, y_array) solution arrays
    
    TODO: Implement scipy.solve_bvp wrapper
    Hint: Set up initial guess and call solve_bvp
    """
    # TODO: Setup initial mesh and guess
    
    # TODO: Call scipy.solve_bvp
    
    # TODO: Extract and return solution
     # Validate input parameters
    if len(x_span) != 2 or x_span[1] <= x_span[0]:
        raise ValueError("x_span must be a tuple (x_start, x_end) with x_end > x_start")
    if len(boundary_conditions) != 2:
        raise ValueError("boundary_conditions must be a tuple (u_left, u_right)")
    if n_points < 5:
        raise ValueError("n_points must be at least 5")
    
    x_start, x_end = x_span
    u_left, u_right = boundary_conditions
    
    # Initial mesh
    x_init = np.linspace(x_start, x_end, n_points)
    
    # Initial guess (linear interpolation between boundary values)
    y_init = np.zeros((2, n_points))
    y_init[0] = np.linspace(u_left, u_right, n_points)
    
    # Solve BVP
    sol = solve_bvp(ode_system_scipy, boundary_conditions_scipy, x_init, y_init, tol=1e-6)
    
    if not sol.success:
        raise RuntimeError(f"scipy.solve_bvp failed: {sol.message}")
    
    # Evaluate solution on fine grid
    x_fine = np.linspace(x_start, x_end, 100)
    y_fine = sol.sol(x_fine)[0]
    
    return x_fine, y_fine


def compare_methods_and_plot(x_span=(0, 1), boundary_conditions=(1, 1), n_points=100):
    """
    Compare shooting method and scipy.solve_bvp, generate comparison plot.
    
    Args:
        x_span (tuple): Domain for the problem
        boundary_conditions (tuple): Boundary values (left, right)
        n_points (int): Number of points for plotting
    
    Returns:
        dict: Dictionary containing solutions and analysis
    
    TODO: Implement method comparison and visualization
    Hint: Call both methods, plot results, calculate differences
    """
    # TODO: Solve using both methods
    
    # TODO: Create comparison plot with English labels
    
    # TODO: Calculate and display differences
    
    # TODO: Return analysis results
    print("Solving BVP using both methods...")
    
    try:
        # Solve using shooting method
        print("Running shooting method...")
        x_shoot, y_shoot = solve_bvp_shooting_method(x_span, boundary_conditions, n_points)
        
        # Solve using scipy.solve_bvp
        print("Running scipy.solve_bvp...")
        x_scipy, y_scipy = solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points//2)
        
        # Interpolate scipy solution to shooting method grid for comparison
        y_scipy_interp = np.interp(x_shoot, x_scipy, y_scipy)
        
        # Calculate differences
        max_diff = np.max(np.abs(y_shoot - y_scipy_interp))
        rms_diff = np.sqrt(np.mean((y_shoot - y_scipy_interp)**2))
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        
        # Main comparison plot
        plt.subplot(2, 1, 1)
        plt.plot(x_shoot, y_shoot, 'b-', linewidth=2, label='Shooting Method')
        plt.plot(x_scipy, y_scipy, 'r--', linewidth=2, label='scipy.solve_bvp')
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.title('Comparison of BVP Solution Methods')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Mark boundary points
        plt.plot([x_span[0], x_span[1]], [boundary_conditions[0], boundary_conditions[1]], 
                'ko', markersize=8, label='Boundary Conditions')
        plt.legend()
        
        # Difference plot
        plt.subplot(2, 1, 2)
        plt.plot(x_shoot, y_shoot - y_scipy_interp, 'g-', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('Difference (Shooting - scipy)')
        plt.title(f'Solution Difference (Max: {max_diff:.2e}, RMS: {rms_diff:.2e})')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print analysis
        print("\nSolution Analysis:")
        print(f"Maximum difference: {max_diff:.2e}")
        print(f"RMS difference: {rms_diff:.2e}")
        print(f"Shooting method points: {len(x_shoot)}")
        print(f"scipy.solve_bvp points: {len(x_scipy)}")
        
        # Verify boundary conditions
        print(f"\nBoundary condition verification:")
        print(f"Shooting method: u({x_span[0]}) = {y_shoot[0]:.6f}, u({x_span[1]}) = {y_shoot[-1]:.6f}")
        print(f"scipy.solve_bvp: u({x_span[0]}) = {y_scipy[0]:.6f}, u({x_span[1]}) = {y_scipy[-1]:.6f}")
        print(f"Target: u({x_span[0]}) = {boundary_conditions[0]}, u({x_span[1]}) = {boundary_conditions[1]}")
        
        return {
            'x_shooting': x_shoot,
            'y_shooting': y_shoot,
            'x_scipy': x_scipy,
            'y_scipy': y_scipy,
            'max_difference': max_diff,
            'rms_difference': rms_diff,
            'boundary_error_shooting': [abs(y_shoot[0] - boundary_conditions[0]), 
                                      abs(y_shoot[-1] - boundary_conditions[1])],
            'boundary_error_scipy': [abs(y_scipy[0] - boundary_conditions[0]), 
                                   abs(y_scipy[-1] - boundary_conditions[1])]
        }
        
    except Exception as e:
        print(f"Error in method comparison: {str(e)}")
        raise


# Test functions for development and debugging
def test_ode_system():
    """
    Test the ODE system implementation.
    """
    print("Testing ODE system...")
    try:
        # Test point
        t_test = 0.5
        y_test = np.array([1.0, 0.5])
        
        # Test shooting method ODE system
        dydt = ode_system_shooting(t_test, y_test)
        print(f"ODE system (shooting): dydt = {dydt}")
        
        # Test scipy ODE system
        dydt_scipy = ode_system_scipy(t_test, y_test)
        print(f"ODE system (scipy): dydt = {dydt_scipy}")
        
    except NotImplementedError:
        print("ODE system functions not yet implemented.")


def test_boundary_conditions():
    """
    Test the boundary conditions implementation.
    """
    print("Testing boundary conditions...")
    try:
        ya = np.array([1.0, 0.5])  # Left boundary
        yb = np.array([1.0, -0.3])  # Right boundary
        
        bc_residual = boundary_conditions_scipy(ya, yb)
        print(f"Boundary condition residuals: {bc_residual}")
        
    except NotImplementedError:
        print("Boundary conditions function not yet implemented.")


if __name__ == "__main__":
    print("项目2：打靶法与scipy.solve_bvp求解边值问题")
    print("=" * 50)
    
    # Run basic tests
    test_ode_system()
    test_boundary_conditions()
    
    # Try to run comparison (will fail until functions are implemented)
    try:
        print("\nTesting method comparison...")
        results = compare_methods_and_plot()
        print("Method comparison completed successfully!")
    except NotImplementedError as e:
        print(f"Method comparison not yet implemented: {e}")
    
    print("\n请实现所有标记为 TODO 的函数以完成项目。")
