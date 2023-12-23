import numpy as np
from network import createNetwork, load_data
import openpyxl as xl
from math import sqrt, pi

def correct_pressures(network: object, params: object, iter: int):
    '''Correct the Pressures of the nodes. Number of iterations is equal to the "iter" parameter.'''
    dP_old = 0

    j = 0
    while True:
        # print(f"Iter: {j}")
        dP = correction_factor(network, params)    

        if (np.all(np.abs(dP - dP_old) <= 1.0e-10)):  # check if the loop has converged
            break
        
        j += 1
        if (iter == 1 and j == iter):
            iter_1(network)
            break
        elif (iter == 2 and j == iter):
            iter_2(network)
            break
                        
        dP_old = dP

    if (iter == None):
        print("\nPressures: ")
        print(network.nodesList[:, 2])  # pressure in the nodes
        print(4 * network.edgesList[:, 7] / (pi * np.power(network.edgesList[:, 3], 2)))  # velocity of the flow for each pipeline, U=(4Q)/(πD^2)
      
def correction_factor(network: object, params: object) -> np.ndarray:
    ''' Computation of Pressure Corrections of the nodes.'''

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
            der = 0.5 * 1/sqrt(edge[6] * 9.81 * params["D2"].value) * 1/sqrt(abs(edge[5]))  # transform pressure from mΣΥ=Pa/(ρ*g) to Pa
            J[i, i] -= der

            if (i == int(edge[0])):
                J[i, int(edge[1])] = der
            else:
                J[i, int(edge[0])] = der

        F[i] += node[3]  # add consumption

    dP = np.linalg.solve(J, -F)  # J*dP + F = 0 -> J*dP = -F
    dP = np.squeeze(dP)
    network.nodesList[:, 2] += dP
    network.edgeAttributes(init=False)  # update the attributes of the pipelines with the new P

    return dP

def iter_1(network):
    '''Iteration 1: Check the direction of the supply for each edge.'''

    edges = network.edgesList

    print("")
    for edge in edges:
        if (edge[5] > 0):  # dP>0 -> Q: nodeA -> nodeB
            print(f"Edge {int(edge[0])+1}-{int(edge[1])+1}: Q=+{edge[7]}")
        else:              # dP<0 -> Q: nodeB -> nodeA
            print(f"Edge {int(edge[0])+1}-{int(edge[1])+1}: Q=-{edge[7]}")

def iter_2(network):
    '''Iteration 2: Selection of standardized diameters.'''

    edges = network.edgesList
    d = np.sqrt((4*edges[:, 7]) / (4*pi))  # d=sqrt(4Q/(U*π)) where U=4 m/s and Q is the supply
    
    for i, edge in enumerate(edges):
        print(f"Edge {int(edge[0])+1}-{int(edge[1])+1}: D={d[i]}")

def main():
    path = "network_data.xlsx"
    nodes = load_data(path, "Nodes")  # access the "Nodes" sheet from the excel file
    edges = load_data(path, "Edges")
    params = xl.load_workbook(path)["Params"]  
    daily_demand = xl.load_workbook(path)["Daily demand"]  
    
    net = createNetwork(nodes, edges, params)
    # net.plotNetwork(plot_arrows=False)
    
    correct_pressures(net, params, 1)  # check the direction of each edge's supply
    # net.plotNetwork(plot_arrows=True)
    print(f"\nOpen file '{path}', sheet'Edges' sheet and update the values of ζ.") 
    next = input(f"Type [yes] and click enter if you updated the values of ζ: ")
    print("")

    if (next == "yes"):
        net.update_zeta(path)  # define the values of ζ for each node
        correct_pressures(net, params, 2)  
    
    next = input(f"\nType [yes] and click enter if you have selected the standardized diameters: ")

    if (next == "yes"):
        net.edgesList[:, 3] = load_data(path, "Edges")[:, 4]  # update diameters
        correct_pressures(net, params, None)

if __name__ == '__main__':
    main()