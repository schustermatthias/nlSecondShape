import nlfem
import numpy as np

import fenics
import meshio


def convert_msh_to_xdmf(name):
    msh = meshio.read("mesh/" + name + ".msh")
    for cell in msh.cells:
        if cell.type == "triangle":
            elements = cell.data
        elif cell.type == "line":
            lines = cell.data

    for key in msh.cell_data_dict["gmsh:physical"].keys():
        if key == "triangle":
            elementLabels = msh.cell_data_dict["gmsh:physical"][key]
        elif key == "line":
            lineLabels = msh.cell_data_dict["gmsh:physical"][key]

    mesh = meshio.Mesh(points=msh.points[:, :2], cells={"triangle": elements})
    mesh_function_subdomains = meshio.Mesh(points=msh.points[:, :2],
                                cells=[("triangle", elements)],
                                cell_data={"name_to_read": [elementLabels]})
    mesh_function_boundaries = meshio.Mesh(points=msh.points[:, :2],
                                           cells=[("line", lines)],
                                           cell_data={"name_to_read": [lineLabels]})
    meshio.write("mesh/" + name + ".xdmf", mesh)
    meshio.write("mesh/" + name + "_subdomains.xdmf", mesh_function_subdomains)
    meshio.write("mesh/" + name + "_boundaries.xdmf", mesh_function_boundaries)


def get_fenics_mesh_and_mesh_data(name, interface_label):
    mesh = fenics.Mesh()
    with fenics.XDMFFile("mesh/" + name + ".xdmf") as infile:
        infile.read(mesh)

    subdomain_function = get_fenics_subdomain_function(mesh, name)
    interface_function = get_fenics_interface_function(mesh, name, interface_label)

    return mesh, subdomain_function, interface_function


def get_fenics_subdomain_function(mesh, name):
    subdomain_data = fenics.MeshValueCollection("size_t", mesh, 2)
    with fenics.XDMFFile("mesh/" + name + "_subdomains.xdmf") as infile:
        infile.read(subdomain_data)
    return fenics.cpp.mesh.MeshFunctionSizet(mesh, subdomain_data)


def get_fenics_interface_function(mesh, name, interface_label):
    boundary_data = fenics.MeshValueCollection("size_t", mesh, 1)
    with fenics.XDMFFile("mesh/" + name + "_boundaries.xdmf") as infile:
        infile.read(boundary_data)
    interface_function = fenics.cpp.mesh.MeshFunctionSizet(mesh, boundary_data)
    interface_array = interface_function.array()
    # set other boundary labels to 0
    for l in range(interface_array.size):
        if interface_array[l] != interface_label:
            interface_function.set_value(l, 0)
    return interface_function


def convert_fenics_mesh_to_dict(mesh, subdomains, nonlocal_boundary_label):
    elements = np.array(mesh.cells(), dtype=int)
    vertices = mesh.coordinates()
    num_elements = elements.shape[0]
    elementLabels = np.array(subdomains.array(), dtype='long')
    for triangle in range(num_elements):
        if elementLabels[triangle] == nonlocal_boundary_label:
            elementLabels[triangle] = -1.0

    vertexLabels = nlfem.get_vertexLabel(elements, elementLabels, vertices)
    vertexLabels_doubled = np.concatenate((vertexLabels, vertexLabels))
    mesh_dict = {"elements": elements, "vertices": vertices, "elementLabels": elementLabels,
                 "vertexLabels": vertexLabels, "vertexLabels_doubled": vertexLabels_doubled}
    return mesh_dict


def get_interface_indices(mesh, interface, interface_label):
    # Find facets on interior boundary
    indices_interface_facets = []
    for facet in range(len(interface)):
        if interface[facet] == interface_label:
            indices_interface_facets.append(facet)

    # Find vertices on interior boundary
    interface_vertices = []
    interface_elements = []
    for cell in fenics.cells(mesh):
        for facet in fenics.facets(cell):
            if facet.index() in indices_interface_facets:
                interface_elements.append(cell.index())
                for vertex in fenics.vertices(facet):
                    interface_vertices.append(vertex.index())

    return list(set(interface_vertices)), list(set(interface_elements))


class MeshData:
    def __init__(self, file_name, interface_label, nonlocal_boundary_label):
        convert_msh_to_xdmf(file_name)
        self.mesh, self.subdomains, self.interface = get_fenics_mesh_and_mesh_data(file_name, interface_label)
        self.mesh_dict = convert_fenics_mesh_to_dict(self.mesh, self.subdomains, nonlocal_boundary_label)
        self.interface_vertices, self.interface_elements = get_interface_indices(self.mesh, self.interface,
                                                                                 interface_label)
        indices_vertices = list(range(self.mesh.num_vertices()))
        self.indices_nodes_not_on_interface = list(set(indices_vertices) - set(self.interface_vertices))

    def update(self, mesh_update):
        fenics.ALE.move(self.mesh, mesh_update)
        self.mesh_dict["vertices"] = self.mesh.coordinates()
