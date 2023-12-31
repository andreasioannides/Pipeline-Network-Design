import numpy as np
from network import createNetwork, load_data
import openpyxl as xl
from math import sqrt, pi

def correct_pressures(network: object, params: object, iter: int) -> np.ndarray:
    '''Correct the Pressures of the nodes.'''

    dP_old = 0
    
    while True:
        dP = correction_factor(network, params)  

        if (np.all(np.abs(dP - dP_old) <= 1.0e-10)):  # check if the loop has converged
            break
        
        dP_old = dP
    
    if (iter == 1):
        iter_1(network)
    elif (iter == 2):
        iter_2(network)
    elif (iter == 3):
        iter_3(network)
    
    return dP_old

def correction_factor(network: object, params: object) -> np.ndarray:
    ''' Computation of Pressure Corrections of the nodes.'''

    num_nodes = len(network.nodesList)  
    bc_nodes = [cell.value for cell in params['D'][1:]]  # list with the indexes of nodes with boundary conditions

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
            der = 0.5 * 1/sqrt(edge[6] * 9.81 * params["C2"].value) * 1/sqrt(abs(edge[5]))  # transform pressure from mΣΥ=Pa/(ρ*g) to Pa
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

def iter_1(network: object):
    '''Iteration 1: Define the direction of the supply for each edge.'''

    print("")
    for i, p in enumerate(network.nodesList[:, 2]): 
        print(f"Node {i+1}: P={p}")

def iter_2(network: object):
    '''Iteration 2: Selection of standardized diameters.'''

    edges = network.edgesList
    r = (4*edges[:, 7]) / (4*pi)
    d = np.sqrt(r.astype(float))  # d=sqrt(4Q/(U*π)) where U=4 m/s and Q is the supply
    
    for i, edge in enumerate(edges):
        print(f"Edge {int(edge[0])+1}-{int(edge[1])+1}: D={d[i]}")

def iter_3(network: object):
    '''Iteration 3: Calculate the pressure in the nodes and the velosity of the flow in the pipelines.'''

    print("\nPressure: ")
    print(network.nodesList[:, 2])  # pressure in the nodes
    print("Velosity: ")
    print(4 * network.edgesList[:, 7] / (pi * np.power(network.edgesList[:, 3], 2)))  # velosity of the flow for each pipeline, U=(4Q)/(πD^2)

def daily(network: object, nodes: np.ndarray, params: object, daily_demand: object):
    '''Calculate the pressures of the nodes during the day.'''

    daily_demand = [cell.value for cell in daily_demand['A'][1:]]
    time = np.arange(0, 24, 2)
    daily_pressure = np.empty(shape=(len(time), len(nodes[:, 2])))

    print("\nNodes with the value of 'False' have pressures < 2500 Pa or velosity > 8 m/s.\n")

    for t, demand in enumerate(daily_demand):
        network.nodesList[:, 2] = nodes[:, 2]
        network.nodesList[:, 3] = nodes[:, 3] * (demand / nodes[0, 3])
        network.edgeAttributes(init=False)
        dp = correct_pressures(network, params, None)
        daily_pressure[t] = network.nodesList[:, 2]
        
        print(f"\nTime {time[t]}:")
        print(network.nodesList[:, 2] > 2500)
        print(4 * network.edgesList[:, 7] / (pi * np.power(network.edgesList[:, 3], 2)) < 8)

    network.plot_P(np.transpose(daily_pressure))

def main():

    '''Load data from excel file.'''
    path = "network_data.xlsx"
    nodes = load_data(path, "Nodes")  # access the "Nodes" sheet from the excel file
    edges = load_data(path, "Edges")
    params = xl.load_workbook(path)["Params"]  
    daily_demand = xl.load_workbook(path)["Daily demand"]  
    
    '''Create the pipeline network.'''
    net = createNetwork(np.copy(nodes), np.copy(edges), params)  # use np.copy to avoid modifying the original array. Arrays are passed by reference inside functions
    # net.plotNetwork()
    
    '''Define the values of ζ for each node.'''
    p = correct_pressures(net, params, 1)  # check the direction of each edge's supply
    
    print(f"\nOpen file '{path}', 'Edges' sheet and update the values of ζ.") 
    next = input(f"Type [yes] and click enter if you updated the values of ζ: ")
    print("")

    if (next == "yes"):
        net.nodesList = np.copy(nodes)
        net.edgesList[:, 8] = load_data(path, "Edges")[:, 3]  # update the values of ζ for each node
        net.edgeAttributes(init=False)
        p = correct_pressures(net, params, 2)  

    '''Select standardized diameters for the pipelines.'''
    next = input(f"\nType [yes] and click enter if you have selected the standardized diameters: ")

    if (next == "yes"):
        net.nodesList = np.copy(nodes)
        net.edgesList[:, 3] = load_data(path, "Edges")[:, 4]  # update diameters
        net.edgeAttributes(init=False)
        p = correct_pressures(net, params, 3)

    '''Calculate the pressures of the nodes during the day.'''
    daily(net, np.copy(nodes), params, daily_demand)

if __name__ == '__main__':
    main()