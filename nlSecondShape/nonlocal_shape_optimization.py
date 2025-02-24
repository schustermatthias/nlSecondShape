import time
import numpy as np
import scipy.sparse
from scipy.sparse.linalg import lgmres
import multiprocessing
import pathos.multiprocessing

import fenics
import nlfem

import helper
import mesh_data
import results


class NonlocalShapeProblem:

    def __init__(self, conf):
        fenics.parameters["reorder_dofs_serial"] = False  # Order of vertices and dofs coincide
        self.nlfem_shape_conf = conf["nlfem_shape_conf"]
        self.nlfem_conf = conf["nlfem_conf"]
        self.kernel = conf["kernel"]
        self.interface_label = conf["interface_label"]
        self.boundary_label = conf["boundary_label"]

        self.source = conf["source"]

        self.target_mesh_data = mesh_data.MeshData(conf['target_shape'], self.interface_label, self.boundary_label)
        self.current_mesh_data = mesh_data.MeshData(conf['init_shape'], self.interface_label, self.boundary_label)

        self.results = results.Results(self.current_mesh_data.subdomains, self.target_mesh_data.interface, conf)

        self.data = helper.compute_data(self.target_mesh_data.mesh, self.target_mesh_data.subdomains, self.source,
                                        self.target_mesh_data.mesh_dict, self.kernel, self.nlfem_conf)
        self.state_1 = helper.initialize_function(self.current_mesh_data)
        self.state_2 = helper.initialize_function(self.current_mesh_data)

        self.c_per = conf["c_per"]
        self.epsilon = conf["epsilon"]

    def compute_mesh_deformation(self, cg_space, cg2_space, dx, ds, normal, state_1, state_2, data, source, mesh_dict,
                                 epsilon, c_per):
        shape_derivative = self.assemble_shape_derivative(cg2_space, dx, ds, normal, state_1, state_2, data, source,
                                                          mesh_dict, c_per)
        # Compute the two adjoints and then assemble the corresponding row of the stiffness matrix used to compute the
        # deformation
        stiffness_matrix = self.assemble_stiffness_matrix(cg_space, cg2_space, dx, ds, normal, state_1, state_2, data,
                                                          source, mesh_dict, epsilon, c_per)
        vector_field, info = lgmres(stiffness_matrix, -1.0 * shape_derivative, tol=1E-8)

        vertexLabels_doubled = mesh_dict["vertexLabels_doubled"]
        mesh_update_vec = np.zeros(len(vertexLabels_doubled))
        mesh_update_vec[vertexLabels_doubled > 0.0] = vector_field
        mesh_update = helper.convert_vec_to_fenics_function(cg2_space, mesh_update_vec)

        return mesh_update, shape_derivative

    def assemble_stiffness_matrix(self, cg_space, cg2_space, dx, ds, normal, state_1, state_2, data, source, mesh_dict,
                                  epsilon, c_per):
        t_1 = time.time()
        elements = mesh_dict["elements"]
        elementLabels = mesh_dict["elementLabels"]
        vertices = mesh_dict["vertices"]
        nlfem_dict = {"state": None, "adjoint": None}
        mesh_nl, A_nl = nlfem.stiffnessMatrix_fromArray(elements, elementLabels, vertices, self.kernel,
                                                        self.nlfem_conf, nlfem_dict)
        vertexLabels = mesh_dict["vertexLabels"]
        A_nl = A_nl[vertexLabels > 0][:, vertexLabels > 0]

        forcing_terms_adjoint_1 = self.assemble_forcing_terms_adjoint_1(cg_space, cg2_space, dx, state_1, source,
                                                                        mesh_dict)
        forcing_terms_adjoint_2_part_1 = self.assemble_forcing_terms_adjoint_2_part_1(cg_space, cg2_space, dx, state_1,
                                                                                      state_2, data, mesh_dict)
        forcing_terms_adjoint_2_part_2 = self.assemble_forcing_terms_adjoint_2_part_2(cg_space, dx, mesh_dict)
        stiffness_matrix = self.assemble_second_derivative_obj_func(cg2_space, dx, ds, normal, state_1, state_2, data,
                                                                    source, mesh_dict, epsilon, c_per)
        dofs_vector_space = np.shape(forcing_terms_adjoint_1)[1]

        # Assemble stiffness_matrix
        num_cores = multiprocessing.cpu_count()

        def assemble_submatrix(index_set):
            submatrix = scipy.sparse.lil_matrix((dofs_vector_space, dofs_vector_space))
            for i in index_set:
                # compute adjoint_1
                b = forcing_terms_adjoint_1[:, i]
                b = -1.0 * b.toarray()
                adjoint_1, info = lgmres(A_nl.transpose(), b, tol=1E-10)

                # compute adjoint_2
                # assemble r.h.s.
                b = -1.0 * forcing_terms_adjoint_2_part_1[:, i].toarray()
                b_2 = forcing_terms_adjoint_2_part_2 @ adjoint_1
                b = b.flatten() - b_2
                adjoint_2, info = lgmres(A_nl, b, tol=1E-10)

                # assemble row
                new_row = adjoint_2 @ forcing_terms_adjoint_1 + adjoint_1 @ forcing_terms_adjoint_2_part_1
                submatrix[i] = new_row
            return submatrix

        indices = list(range(dofs_vector_space))
        index_sets = np.array_split(indices, dofs_vector_space)
        pool = pathos.multiprocessing.ProcessingPool(num_cores)
        submatrices = pool.map(assemble_submatrix, index_sets)
        pool.close()
        pool.join()
        pool.clear()

        for i in range(num_cores):
            submatrix = submatrices[i].tocsr()
            stiffness_matrix += submatrix
        t_2 = time.time()
        print("Implementation of stiffness matrix finished. Time needed: " + str(t_2 - t_1))
        return stiffness_matrix

    def assemble_forcing_terms_adjoint_1(self, cg_space, cg2_space, dx, state, source, mesh_dict):
        v_test = fenics.TestFunction(cg_space)
        W_trial = fenics.TrialFunction(cg2_space)

        # Assemble single integrals with fenics
        a = source[0] * v_test * fenics.div(W_trial) * dx(1) + source[1] * v_test * fenics.div(W_trial) * dx(2)

        A = helper.assemble_matrix(a)
        u_vec = state.vector().get_local()
        nlfem_dict = {"state": u_vec, "adjoint": None}
        self.nlfem_shape_conf['ShapeDerivative'] = 2

        elements = mesh_dict["elements"]
        elementLabels = mesh_dict["elementLabels"]
        vertices = mesh_dict["vertices"]

        mesh_nl, A_nl = nlfem.stiffnessMatrix_fromArray(elements, elementLabels, vertices, self.kernel,
                                                        self.nlfem_shape_conf, nlfem_dict)

        vertexLabels = mesh_dict["vertexLabels"]
        vertexLabels_doubled = mesh_dict["vertexLabels_doubled"]
        A = A_nl - A

        # A_res = A[vertexLabels > 0][:, vertexLabels_doubled < 0]
        A = A[vertexLabels > 0][:, vertexLabels_doubled > 0]
        return A #, A_res

    def assemble_forcing_terms_adjoint_2_part_1(self, cg_space, cg2_space, dx, state_1, state_2, data, mesh_dict):
        u_trial = fenics.TrialFunction(cg_space)
        W_test = fenics.TestFunction(cg2_space)

        # Assemble single integrals with fenics
        a = (state_1 - data) * u_trial * fenics.div(W_test) * dx - u_trial * fenics.dot(fenics.grad(data), W_test) * dx
        A = helper.assemble_matrix(a)

        # Assemble double integrals with shape version of nlfem
        v_vec = state_2.vector().get_local()
        nlfem_dict = {"state": None, "adjoint": v_vec}

        self.nlfem_shape_conf['ShapeDerivative'] = 2

        elements = mesh_dict["elements"]
        elementLabels = mesh_dict["elementLabels"]
        vertices = mesh_dict["vertices"]

        mesh_nl, A_nl = nlfem.stiffnessMatrix_fromArray(elements, elementLabels, vertices, self.kernel,
                                                        self.nlfem_shape_conf, nlfem_dict)

        vertexLabels = mesh_dict["vertexLabels"]
        vertexLabels_doubled = mesh_dict["vertexLabels_doubled"]
        A = A.transpose() + A_nl

        #  A_res = A[vertexLabels > 0][:, vertexLabels_doubled < 0]
        A = A[vertexLabels > 0][:, vertexLabels_doubled > 0]

        return A  #, A_res

    def assemble_forcing_terms_adjoint_2_part_2(self, cg_space, dx, mesh_dict):
        u_trial = fenics.TrialFunction(cg_space)
        v_test = fenics.TestFunction(cg_space)
        a = fenics.dot(u_trial, v_test) * dx

        A = helper.assemble_matrix(a)

        vertexLabels = mesh_dict["vertexLabels"]

        # A_res = A[vertexLabels > 0][:, vertexLabels < 0]
        A = A[vertexLabels > 0][:, vertexLabels > 0]

        return A  # , A_res

    def assemble_second_derivative_obj_func(self, cg2_space, dx, ds, normal, state, adjoint, data, source, mesh_dict,
                                            epsilon, c_per):
        V_trial = fenics.TrialFunction(cg2_space)
        V_test = fenics.TestFunction(cg2_space)

        DV_test = fenics.grad(V_test)
        DV_trial = fenics.grad(V_trial)

        dmdata_trial = fenics.dot(fenics.grad(data), V_trial)
        dmdata_test = fenics.dot(fenics.grad(data), V_test)

        a_1 = 0.5 * (state - data)**2 * (fenics.div(V_test) * fenics.div(V_trial)
                                     - fenics.tr(fenics.dot(DV_test, DV_trial))) * dx
        a_2 = (- (state - data) * dmdata_test * fenics.div(V_trial) * dx
               - (state - data) * dmdata_trial * fenics.div(V_test) * dx
               + dmdata_trial * dmdata_test * dx)
        a_3 = (source[0] * adjoint * (fenics.div(V_test)*fenics.div(V_trial) - fenics.tr(fenics.dot(DV_test, DV_trial)))*dx(1)
            + source[1] * adjoint * (fenics.div(V_test)*fenics.div(V_trial) - fenics.tr(fenics.dot(DV_test, DV_trial)))*dx(2))

        a_4 = (epsilon * fenics.inner(V_trial, V_test) + epsilon * fenics.inner(DV_trial, DV_test))*dx
        a_5 = helper.get_perimeter_hessian(V_test, V_trial, normal, ds, self.interface_label)

        a = a_1 + a_2 - a_3 + a_4 + c_per*a_5
        A = helper.assemble_matrix(a)

        state_vec = state.vector().get_local()
        adjoint_vec = adjoint.vector().get_local()
        nlfem_dict = {"state": state_vec, "adjoint": adjoint_vec}

        elements = mesh_dict["elements"]
        elementLabels = mesh_dict["elementLabels"]
        vertices = mesh_dict["vertices"]
        self.nlfem_shape_conf['ShapeDerivative'] = 2
        mesh_nl, A_nl = nlfem.stiffnessMatrix_fromArray(elements, elementLabels, vertices, self.kernel,
                                                        self.nlfem_shape_conf, nlfem_dict)

        A = A + A_nl

        vertexLabels_doubled = mesh_dict["vertexLabels_doubled"]

        # A_res = A[vertexLabels_doubled > 0][:, vertexLabels_doubled < 0]
        A = A[vertexLabels_doubled > 0][:, vertexLabels_doubled > 0]

        return A  # , A_res

    def assemble_shape_derivative(self, cg2_space, dx, ds, normal, state_1, state_2, data, source, mesh_dict, c_per):
        V_test = fenics.TestFunction(cg2_space)

        a_1 = (fenics.div(V_test) * 0.5 * (state_1 - data)**2
               - (state_1 - data) * fenics.dot(fenics.grad(data), V_test))*dx
        a_2 = fenics.div(V_test) * source[0] * state_2*dx(1) + fenics.div(V_test) * source[1] * state_2*dx(2)

        a_3 = helper.get_perimeter_derivative(V_test, normal, ds, self.interface_label)
        a = a_1 - a_2 + c_per * a_3

        b = helper.assemble_vector(a)

        state_1_vec = state_1.vector().get_local()
        state_2_vec = state_2.vector().get_local()
        nlfem_dict = {"state": state_1_vec, "adjoint": state_2_vec}

        elements = mesh_dict["elements"]
        elementLabels = mesh_dict["elementLabels"]
        vertices = mesh_dict["vertices"]
        self.nlfem_shape_conf['ShapeDerivative'] = 1
        mesh_nl, b_nl = nlfem.stiffnessMatrix_fromArray(elements, elementLabels, vertices, self.kernel,
                                                          self.nlfem_shape_conf, nlfem_dict)
        b_nl = b_nl.toarray().flatten()
        b = b + b_nl

        number_nodes = len(mesh_dict["vertexLabels"])
        for index in self.current_mesh_data.indices_nodes_not_on_interface:
            b[index] = 0.0
            b[index + number_nodes] = 0.0

        vertexLabels_doubled = mesh_dict["vertexLabels_doubled"]
        b = b[vertexLabels_doubled > 0]

        return b

    def solve(self, max_iterations, deformation_tol=None):
        deformation_norm = helper.initialize_shape_gradient_norm(deformation_tol)
        k = 1
        epsilon = self.epsilon
        while helper.iteration_continues(k, max_iterations, deformation_norm, deformation_tol):
            print("Iteration " + str(k) + ":")
            self.results.save_subdomains(self.current_mesh_data.subdomains)

            # Get mesh information
            mesh = self.current_mesh_data.mesh
            mesh_dict = self.current_mesh_data.mesh_dict
            subdomains = self.current_mesh_data.subdomains
            interface = self.current_mesh_data.interface
            cg_space = fenics.FunctionSpace(mesh, 'CG', 1)
            cg2_space = fenics.VectorFunctionSpace(mesh, 'CG', 1)
            dx = fenics.Measure('dx', domain=mesh, subdomain_data=subdomains)
            ds = fenics.Measure('dS', domain=mesh, subdomain_data=interface)
            normal = fenics.FacetNormal(self.current_mesh_data.mesh)

            # project data onto the current mesh
            projected_data = fenics.project(self.data, cg_space)

            # Compute state_1
            self.state_1 = helper.solve_state_1(mesh, subdomains, self.source, mesh_dict, self.kernel, self.nlfem_conf)
            # Compute state_2
            self.state_2 = helper.solve_state_2(mesh, self.state_1, self.data, mesh_dict, self.kernel, self.nlfem_conf)
            self.results.save_state(self.state_1)
            # self.results.save_adjoint(self.state_2)

            # Compute mesh deformation
            mesh_update, shape_derivative = self.compute_mesh_deformation(cg_space, cg2_space, dx, ds, normal,
                                                                          self.state_1, self.state_2, projected_data,
                                                                          self.source, mesh_dict, epsilon,
                                                                          self.c_per)

            deformation_norm = helper.get_shape_gradient_norm(self.current_mesh_data.mesh, mesh_update)
            objective_function_value = helper.get_target_function_value(self.current_mesh_data, self.state_1,
                                                                        projected_data, self.c_per)
            fenics_shape_derivative = helper.convert_vec_to_fenics_function(cg2_space, shape_derivative)

            self.results.store_additional_results(self.current_mesh_data.mesh, mesh_update, objective_function_value,
                                                  fenics_shape_derivative)
            self.current_mesh_data.update(mesh_update)

            k = k + 1

        # save final state, adjoint and objective function value
        self.state_1 = helper.solve_state_1(self.current_mesh_data.mesh, self.current_mesh_data.subdomains, self.source,
                                            self.current_mesh_data.mesh_dict, self.kernel, self.nlfem_conf)
        self.state_2 = helper.solve_state_2(self.current_mesh_data.mesh, self.state_1, self.data,
                                            self.current_mesh_data.mesh_dict, self.kernel, self.nlfem_conf)
        self.results.save_results(self.state_1, self.state_2, self.current_mesh_data.subdomains)
        self.results.compute_and_add_objective_function_value(self.data, self.current_mesh_data, self.state_1,
                                                              self.c_per)
        self.results.plot_and_save_additional_results()
