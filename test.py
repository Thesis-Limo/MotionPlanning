import CubicSpline.cubic_spline_planner as cubic_spline_planner
import FrenetOptimalTrajectory.frenet_optimal_trajectory as frenet_optimal_trajectory
import QuinticPolynomialsPlanner.quintic_polynomials_planner as quintic_polynomials_planner

# Test Cubic Spline Planners
cubic_spline_planner.main_1d()
cubic_spline_planner.main_2d()

# Test Quintic Polynomial Planner
quintic_polynomials_planner.main()

# Test Frenet Optimal Trajectory Planner
frenet_optimal_trajectory.main()
