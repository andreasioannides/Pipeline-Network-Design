import numpy as np
from network import createNetwork, load_data
import openpyxl as xl
from math import sqrt, pi

def iterations(network: object, params: object, iter: int):
    dP_old = 0

    j = 0
    while True:
        # print(f"Iter: {j}")
        dP = correct_pressures(network, params)    

        if (np.all(np.abs(dP - dP_old) <= 1.0e-10)):  # check if the loop has converged
            break
        
        j += 1

        if (j == iter):
            # print(network.edgesList[:, 5])
            iter_1(network)
            break
        #     elif (iter == 2):
                        
        dP_old = dP

    print(network.nodesList[:, 2])  

def correct_pressures(network: object, params: object) -> np.ndarray:
    '''Correct the pressures of the nodes.'''

    num_nodes = len(network.nodesList)  
    bc_nodes = [cell.value for cell in params['E'][1:]]  # list with the indexes of nodes with boundary conditions

    J = np.zeros(shape=[num_nodes, num_nodes], dtype=float)  
    F = np.zeros(shape=[num_nodes, 1], dtype=float)  
    der = 0  # initialize the derivative of Q with respect to h
    
    for i, node in enumerate(network.nodesList):
        if (i+1 in bc_nodes):  # check if the node has boundary conditions
            J[i, i] = 1
            continue

        for edge_idx in network.connectedEdges[i]:  
            edge = network.edgesList[edge_idx]         
            F[i] += network.sign_Q(edge_idx, i)
            der = 0.5 * 1/sqrt(edge[6]* 9.81 * params["D2"].value) * 1/sqrt(abs(edge[5]))  # transform pressure from mΣΥ=Pa/(ρ*g) to Pa
            J[i, i] -= der

            if (i == int(edge[0])):
                J[i, int(edge[1])] = der
            else:
                J[i, int(edge[0])] = der

        F[i] += node[3]  # add consumption

    dP = np.linalg.solve(J, -F)  # J*dP + F = 0 -> J*dP = -F
    dP = np.squeeze(dP)
    network.nodesList[:, 2] += dP
    network.edgeAttributes(init=False)

    return dP

def iter_1(network):
    '''Iteration 1: check the direction of each edge's supply.'''

    edges = network.edgesList

    for edge in edges:
        if (edge[5] > 0):  # dP>0 -> Q: nodeA -> nodeB
            print(f"\nEdge {int(edge[0])}-{int(edge[1])}: +{edge[7]}")
        else:  # dP<0 -> Q: nodeB -> nodeA
            print(f"Edge {int(edge[0])}-{int(edge[1])}: -{edge[7]}")

def main():
    path = "network_data.xlsx"
    nodes = load_data(path, "Nodes")  # access the "Nodes" sheet from the excel file
    edges = load_data(path, "Edges")
    params = xl.load_workbook(path)["Params"]  
    
    net = createNetwork(nodes, edges, params)
    # net.plotNetwork()
    
    iterations(net, params, 1)  # check the direction of each edge's supply
    print(f"\nOpen file: '{path}', sheet'Edges' sheet and update the values of ζ.") 
    next = input(f"Type [yes] and click enter if you updated the values of ζ: ")

    if (next == "yes"):
        net.update_zeta(path)  # define the values of ζ for each node
        # iterations(net, params, 2)  # 
        # iterations(net, params, None)

if __name__ == '__main__':
    main()