import os
import datetime
import matplotlib
import matplotlib.pyplot as plt

import fenics

import helper


def save_dict(dictionary, name):
    file = open(name + ".txt", "a")
    for key in dictionary.keys():
        file.write(str(key) + ": " + str(dictionary[key]) + "\n")
    file.close()


def surface_plot(filename, solution, vertices, cells):
    figure = plt.figure()
    ax = figure.add_subplot(projection='3d', title="Solution u")
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], cells, solution, cmap=matplotlib.colormaps["turbo"],
                    antialiased=False)
    plt.savefig(filename)
    plt.show()


class Results:
    def __init__(self, subdomains, target_interface, conf):
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d") + "/" + datetime.datetime.now().strftime("%H_%M")
        self.results_folder = "results/" + timestamp + "/"

        self.file_state = fenics.File(self.results_folder + "state.pvd")
        self.file_adjoint = fenics.File(self.results_folder + "adjoint.pvd")

        self.file_target_interface = fenics.File(self.results_folder + "target_interface.pvd")
        self.file_subdomains = fenics.File(self.results_folder + "subdomains.pvd")

        subdomains.rename("subdomains", "")
        target_interface.rename("target_interface", "")

        # self.file_subdomains << subdomains
        self.file_target_interface << target_interface

        self.shape_derivative_norm_history = []
        self.deformation_norm_history = []
        self.objective_function_value_history = []

        save_dict(conf, self.results_folder + "configuration")

        self.counter = 1

    def save_state(self, state):
        state.rename("state", "")
        self.file_state << state

    def save_adjoint(self, adjoint):
        adjoint.rename("adjoint", "")
        self.file_adjoint << adjoint

    def save_subdomains(self, subdomains):
        subdomains.rename("subdomains", "")
        self.file_subdomains << subdomains

    def save_results(self, state, adjoint, subdomains):
        self.save_state(state)
        self.save_adjoint(adjoint)
        self.save_subdomains(subdomains)

    def store_additional_results(self, mesh, deformation, objective_function_value, shape_derivative):
        #shape_gradient_norm = fenics.norm(shape_gradient, "L2", mesh)
        shape_derivative_norm = fenics.norm(shape_derivative, "L2", mesh)
        deformation_norm = fenics.norm(deformation, "L2", mesh)

        self.shape_derivative_norm_history.append(shape_derivative_norm)
        self.deformation_norm_history.append(deformation_norm)
        self.objective_function_value_history.append(objective_function_value)

        # fenics.plot(deformation)
        # plt.savefig(self.results_folder + 'deformation_' + str(self.counter))
        # plt.clf()
        # plt.close()

        # fenics.plot(shape_derivative)
        # plt.savefig(self.results_folder + 'shape_derivative_' + str(self.counter))
        # plt.clf()
        # plt.close()
        # self.counter += 1

    def compute_and_add_objective_function_value(self, data_function, current_mesh_data, state_1, c_per):
        cg_space = fenics.FunctionSpace(current_mesh_data.mesh, 'CG', 1)
        projected_data = fenics.project(data_function, cg_space)
        objective_function_value = helper.get_target_function_value(current_mesh_data, state_1, projected_data, c_per)
        self.objective_function_value_history.append(objective_function_value)

    def plot_and_save_additional_results(self, time=None):
        fig = []
        axs = []
        for index in range(3):
            fig.append(plt.figure())
            axs.append(fig[index].add_subplot(111))
            axs[index].set_xlabel('iteration')
            axs[index].set_yscale('log')

        axs[0].plot(self.shape_derivative_norm_history, label="shape derivative norm")
        axs[1].plot(self.deformation_norm_history, label="deformation norm")
        axs[2].plot(self.objective_function_value_history, label="objective function value")

        os.system("mkdir " + self.results_folder + "additional_results")
        names = ["shape_derivative_norm", "deformation_norm", "objective_function_value"]
        for index in range(3):
            axs[index].legend()
            fig[index].savefig(self.results_folder + "additional_results/" + names[index] + ".pdf")
        # plt.show()

        file_additional_results = open(self.results_folder + "additional_results.txt", "a")
        file_additional_results.write("shape derivative norm: " + str(self.shape_derivative_norm_history) + "\n")
        file_additional_results.write("deformation norm: " + str(self.deformation_norm_history) + "\n")
        file_additional_results.write("objective function value: " + str(self.objective_function_value_history) + "\n")
        if time is not None:
            file_additional_results.write("Algorithm ended after " + str(time) + "s")

        file_additional_results.close()
