import openpyxl as xl
import numpy as np
from network import createNetwork
import json
from math import sqrt

# Initialize Global Parameters
with open('params.json', 'r') as file:
    params = json.load(file)

def load_data(path: str, sheet_name: str) -> np.ndarray:
    '''Load the data from excel.'''

    wb = xl.load_workbook(path)
    data = []

    for row in wb[sheet_name].iter_rows(min_row=2, values_only=True):
        data.append(list(row))

    data = np.array(data)

    return data

def correct_pressures(network: object):
    '''Correct the pressures of the nodes given the initial pressures.'''

    num_nodes = len(network.nodesList)
    bc_nodes = params["boundary_conditions_nodes"]  # list with the indexes of nodes with boundary conditions
    nodesList = network.nodesList
    J = np.zeros(shape=[num_nodes, num_nodes], dtype=float)  
    F = np.zeros(shape=[num_nodes, 1], dtype=float)  
    der = 0  # initialize the derivative of Q with respect to h
    p_old = nodesList[:, 2]  # pressures
    edge = 0

    while True:
        for i, node in enumerate(nodesList):
            if (i+1 in bc_nodes):  # check if the node has boundary conditions
                J[i, i] = 1
                continue

            for edge_idx in network.connectedEdges:  
                edge = network.edgesList[edge_idx]
                print(edge[0])         
                F[i] += network.sign_Q(edge_idx, i)
                der = 0.5 * 1/sqrt(edge[6]) * 1/sqrt(abs(edge[5]))
                J[i, i] -= der

                if (i == int(edge[0])):
                    J[i, int(edge[1])] = der
                    # J[i, network.nodesList.index(edge.nodeB)] = der
                else:
                    # J[i, network.nodesList.index(edge.nodeA)] = der
                    J[i, int(edge[0])] = der

            F[i] += node[3]  # add consumption

        dP = np.linalg.solve(J, F)
        p_new = p_old + dP

        if (((p_new-p_old)/p_new).all() <= 1.0e-7):  # check if the loop has converged
            break

        p_old = p_new

        print(J)
        break
        
def main():
    path = "network_data.xlsx"
    nodes = load_data(path, "Nodes")
    edges = load_data(path, "Edges")
    
    net = createNetwork(nodes, edges)
    # net.plotNetwork()
    
    correct_pressures(net)
    # J, F = correct_pressures(net)
    # print(J)

if __name__ == '__main__':
    main()