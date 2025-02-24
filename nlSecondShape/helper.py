import time

import scipy
import numpy as np
from scipy.sparse.linalg import gmres

import fenics

import nlfem

def petsc_to_csr_scipy_matrix(petsc_matrix):
    indptr, indices, data = petsc_matrix.mat().getValuesCSR()
    n = petsc_matrix.size(0)
    m = petsc_matrix.size(1)
    csr_matrix = scipy.sparse.csr_matrix((data, indices, indptr), shape=(n, m))
    csr_matrix.eliminate_zeros()
    return csr_matrix


def assemble_matrix(a):
    A = fenics.PETScMatrix()
    fenics.assemble(a, tensor=A)
    A = petsc_to_csr_scipy_matrix(A)
    return A


def assemble_vector(a):
    b = fenics.PETScVector()
    fenics.assemble(a, tensor=b)
    b = b.get_local()

    return b


def convert_nonlocal_function_into_fenics_function_cg(mesh, u, theta, vertexLabels):
    V = fenics.FunctionSpace(mesh, 'CG', 1)
    num_vertices = mesh.num_vertices()
    u_sd_temp = np.zeros(num_vertices, dtype=float)
    indices_theta = 0
    indices_u = 0
    for i in range(0, num_vertices):
        if vertexLabels[i] > 0:
            u_sd_temp[i] = u[indices_u]
            indices_u += 1
        else:
            u_sd_temp[i] = theta[indices_theta]
            indices_theta += 1
    u_sd = fenics.Function(V)
    u_sd.vector().set_local(u_sd_temp)
    return u_sd


def compute_data(mesh, subdomains, source, mesh_dict, kernel, nlfem_conf):
    return solve_state_1(mesh, subdomains, source, mesh_dict, kernel, nlfem_conf)


def solve_state_1(mesh, subdomains, source, mesh_dict, kernel, nlfem_conf):
    elements = mesh_dict["elements"]
    elementLabels = mesh_dict["elementLabels"]
    vertices = mesh_dict["vertices"]

    empty_dict = {}
    nlfem_conf['ShapeDerivative'] = 0
    mesh_nl, A = nlfem.stiffnessMatrix_fromArray(elements, elementLabels, vertices, kernel, nlfem_conf,
                                                 empty_dict)

    vertexLabels = mesh_nl["vertexLabels"]

    cg_space = fenics.FunctionSpace(mesh, 'CG', 1)
    v_test = fenics.TestFunction(cg_space)
    dx = fenics.Measure('dx', domain=mesh, subdomain_data=subdomains)
    l = source[0] * v_test * dx(1) + source[1] * v_test * dx(2) + 0 * v_test * dx(3)
    b = fenics.PETScVector()
    fenics.assemble(l, tensor=b)
    b_vec = b.get_local()

    A_O = A[vertexLabels > 0][:, vertexLabels > 0]
    A_I = A[vertexLabels > 0][:, vertexLabels < 0]
    f_O = b_vec[vertexLabels > 0]

    def g2(x):
        return 0.0

    g = np.apply_along_axis(g2, 1, vertices[vertexLabels < 0])
    f_O -= A_I @ g.ravel()
    # u = spsolve(A_O, f_O)
    u, info = gmres(A_O, f_O, f_O, tol=1E-10)

    if info != 0:
        print('Computing of state equation failed with code ' + str(info))
        raise RuntimeError
    u_fenics = convert_nonlocal_function_into_fenics_function_cg(mesh, u, g, vertexLabels)
    return u_fenics


def solve_state_2(mesh, state, data, mesh_dict, kernel, nlfem_conf):
    elements = mesh_dict["elements"]
    vertices = mesh_dict["vertices"]
    elementLabels = mesh_dict["elementLabels"]

    empty_dict = {}
    nlfem_conf['is_ShapeDerivative'] = 0
    mesh_nl, A = nlfem.stiffnessMatrix_fromArray(elements, elementLabels, vertices, kernel, nlfem_conf,
                                                 empty_dict)
    vertexLabels = mesh_nl["vertexLabels"]

    cg_space = fenics.FunctionSpace(mesh, 'CG', 1)
    u_test = fenics.TestFunction(cg_space)
    dx = fenics.Measure('dx', domain=mesh)
    l = -(state - data) * u_test * dx
    b = fenics.PETScVector()
    b = fenics.assemble(l, tensor=b)
    b_vec = b.get_local()

    A_O = A[vertexLabels > 0][:, vertexLabels > 0]
    A_I = A[vertexLabels > 0][:, vertexLabels < 0]
    f_O = b_vec[vertexLabels > 0]

    def g2(x):
        return 0.0

    g = np.apply_along_axis(g2, 1, vertices[vertexLabels < 0])
    f_O -= A_I @ g.ravel()
    v, info = gmres(A_O.transpose(), f_O, f_O, tol=1E-10)
    if info != 0:
        print('Computing of adjoint equation failed with code ' + str(info))
        raise RuntimeError
    v_fenics = convert_nonlocal_function_into_fenics_function_cg(mesh, v, g, vertexLabels)

    return v_fenics


def initialize_function(mesh_data):
    num_vertices = mesh_data.mesh.num_vertices()
    function_data = np.zeros(num_vertices)

    cg_space = fenics.FunctionSpace(mesh_data.mesh, 'CG', 1)
    function = fenics.Function(cg_space)
    function.vector().set_local(function_data)
    return function


def convert_vec_to_fenics_function(cg_space, vec):
    fenics_function = fenics.Function(cg_space)
    fenics_function.vector().set_local(vec)
    return fenics_function


def div_tangential(vector_field, normal):
    return (fenics.div(vector_field("+"))
            - fenics.dot(fenics.dot(fenics.grad(vector_field("+")), normal("+")), normal("+")))


def grad_tangential(vector_field, normal):
    return (fenics.grad(vector_field("+"))
            - fenics.outer(fenics.dot(fenics.grad(vector_field("+")), normal("+")), normal("+")))


def get_perimeter_derivative(vector_field, normal, ds, interface_label):
    return (div_tangential(vector_field, normal)) * ds(interface_label)


def get_perimeter_hessian(vector_field_test, vector_field_trial, normal, ds, interface_label):
    div_test = div_tangential(vector_field_test, normal)
    div_trial = div_tangential(vector_field_trial, normal)

    grad_test = grad_tangential(vector_field_test, normal)
    grad_trial = grad_tangential(vector_field_trial, normal)

    return (div_test * div_trial - fenics.tr(grad_test * grad_trial)
            + fenics.dot(fenics.dot(normal("+"), grad_test), fenics.dot(normal("+"), grad_trial))) * ds(interface_label)


def iteration_continues(iteration, max_iterations, gradient_norm, gradient_tol):
    if iteration <= max_iterations:
        criterion_1 = 1
    else:
        criterion_1 = 0

    if gradient_tol is None:
        criterion_2 = 1
    elif gradient_norm >= gradient_tol:
        criterion_2 = 1
    else:
        criterion_2 = 0

    return criterion_1 * criterion_2


def initialize_shape_gradient_norm(gradient_tol):
    if gradient_tol is None:
        return None
    else:
        return gradient_tol + 1.0


def get_shape_gradient_norm(mesh, shape_gradient):
    shape_gradient_norm = fenics.norm(shape_gradient, "L2", mesh)
    return shape_gradient_norm


def get_target_function_value(mesh_data, state, u_bar, c_per=None):
    V_new = fenics.FunctionSpace(mesh_data.mesh, 'CG', 1)

    value = 1. / 2. * fenics.norm(fenics.project(state - u_bar, V_new), "L2", mesh_data.mesh) ** 2

    if c_per:
        ones = fenics.Function(V_new)
        ones.vector()[:] = 1.0
        # dx = fenics.Measure('dx', domain=mesh_data.mesh, subdomain_data=mesh_data.subdomains)
        ds = fenics.Measure("dS", domain=mesh_data.mesh, subdomain_data=mesh_data.interface)
        regularization_value = c_per * fenics.assemble(ones('+') * ds(12))
        value += regularization_value
    return value
