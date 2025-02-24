import nonlocal_shape_optimization
from conf_fractional import configuration as conf_frac
from conf_int import configuration as conf_int

if __name__ == '__main__':
    max_iteration = 50
    problem = nonlocal_shape_optimization.NonlocalShapeProblem(conf_frac)
    problem.solve(max_iteration, deformation_tol=5E-5)

    # problem_2 = nonlocal_shape_optimization.NonlocalShapeProblem(conf_int)
    # problem_2.solve(max_iteration, deformation_tol=5E-5)
