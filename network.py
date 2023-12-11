import numpy as np
from math import sqrt, pi, log10
import json
import matplotlib.pyplot as plt

# Initialize Global Parameters
with open('params.json', 'r') as file:
    params = json.load(file)

class createNetwork:
    '''
    A pipeline network with nodes and pipelines (called edges).

    Attributes
    ----------
    nodesList : np.ndarray
        Array with the all nodes of the network.
    edgesList : np.ndarray
        Array with all edges of the network.
    connectedEdges : np.ndarray
        Array with the connected edges for each node.

    Methods
    -------
    sign_Q(edge_idx, ref_node_idx)
        Get the supply of an edge with the right sign given the reference node.
    neighborNodes(node_idx)
        Get a list with the indexes of the neighbor nodes of a given node.
    plotNetwork()
        Plot the pipeline network.
    '''

    def __init__(self, nodes: np.ndarray, edges: np.ndarray):
        '''Default network constructor.'''

        self.nodesList = nodes 
        '''
        Array with the all nodes of the network. 
        
        Each node has 4 columns: 
        0: x coordinate
        1: y coordinate
        2: P (pressure of the node)
        3: Q (consumption in the node)
        '''

        z = np.zeros(shape=(edges.shape[0], 5))
        self.edgesList = np.concatenate((edges, z), axis=1) 
        '''
        Array with all edges of the network. 

        Each edge has 8 columns:
        0: index of node A
        1: index of node B
        2: L (lenght)
        3: D (diameter)
        4: epsilon (roughness)
        5: dP (pressure difference)
        6: k (coefficient of linear losses)
        7: Q (supply)
        '''

        self.edgeAttributes(init=True)  
        self.connectedEdges = self.get_connected_edges()
        '''Array with the connected edges for each node.'''

    def edgeAttributes(self, init: bool):
        '''Calculate the attributes of the network's edges.'''

        if (init):
            self.edgesList[:, 4] = params["epsilon"]
            self.edgesList[:, 3] = params["D"]
            self.edgesList[:, :2] -= 1  # reduce the index of the nodes by 1 for better usability with array indexing

        nodeA_idx = np.int32(self.edgesList[:, 0])  # data type casting beacuse the type of nodesList array is float
        nodeB_idx = np.int32(self.edgesList[:, 1])
        self.edgesList[:, 5] = self.nodesList[nodeA_idx, 2] - self.nodesList[nodeB_idx, 2]  # dP = P_A - P_B
        self.edgesList[:, 6] = self.k_coefficient()
        self.edgesList[:, 7] = self.supply()

    def get_connected_edges(self) -> list:
        '''Create an array with the connected edges for each node.'''

        con = []  
        
        for i in range(len(self.nodesList)):
            con.append([])

        for i, edge in enumerate(self.edgesList):
            nodeA_idx = int(edge[0]) 
            nodeB_idx = int(edge[1])
            con[nodeA_idx].append(i)
            con[nodeB_idx].append(i)

        return con
    
    def k_coefficient(self) -> np.ndarray:
        '''Calculate the k coefficient of an edge.'''

        Re = 1.0e10  # Reynolds: initial value set to infinite
        edges = self.edgesList
        lambda_ = np.power(1 / (1.14 - 2*np.log10(edges[:, 4]/edges[:, 3] + 21.25/Re**0.9)), 2)  # Jain equation
        k = lambda_ * (edges[:, 2]/edges[:, 3] * 8 / (pi**2 * 9.81 * np.power(edges[:, 3], 4)))

        return k
    
    # def k_coefficient(self) -> np.ndarray:
    #     '''Calculate the k coefficient of the edge.'''

    #     Re = 1.0e10  # Reynolds: initial value set to infinite
    #     k_old = 0  # random initial value
    #     edges = self.edgesList

    #     while True:
    #         lambda_ = np.power(1 / (1.14 - 2*np.log10(edges[:, 4]/edges[:, 3] + 21.25/Re**0.9)), 2)  # Jain equation
    #         k = lambda_ * (edges[:, 2]/edges[:, 3] * 8 / (pi**2 * 9.81 * np.power(edges[:, 3], 4)))
    #         Q = self.supply()
    #         U = 4*Q / (pi * np.power(edges[:, 3], 2))  # velosity
    #         Re = U * edges[:, 3] / params["viscosity"]

    #         if (np.all(np.abs((k - k_old)/k) <= 1.0e-7)):  # check if the loop has converged
    #             break

    #         k_old = k

    #     return k

    def supply(self) -> np.ndarray:
        '''Calculate the supply Q of an edge.'''

        edges = self.edgesList

        return np.sqrt(np.abs(edges[:, 5]) / (edges[:, 6] * 9.81 * params["density"]))  # To transform pressure from mΣΥ=Pa/(ρ*g) to Pa multiply denominator with ρ*g . Q=(|dP|/(K_AB*ρ*g)^0.5
    
    def sign_Q(self, edge_idx: int, ref_node_idx: int) -> float:
        '''
        Get the supply of an edge with the right sign given the reference node.
        
        Parameters
        ----------
        edge : index of the edge to calculate the supply
        ref_node : index of the reference node in the edge
        '''

        sign = 1
        if (ref_node_idx == int(self.edgesList[edge_idx, 0]) and self.edgesList[edge_idx, 5] > 0):  # check if ref_node is equal with the nodeA
            sign = -1
        elif (ref_node_idx == int(self.edgesList[edge_idx, 1]) and self.edgesList[edge_idx, 5] < 0):  # check if ref_node is equal with the nodeB
            sign = -1     
        
        return sign * self.edgesList[edge_idx, 7]
    
    def neighborNodes(self, node_idx: int) -> list:
        '''
        Get a list with the indexes of the neighbor nodes of a given node.
        
        Parameters
        ----------
        node : index of the node to find its neighbors
        '''

        neigh_nodes = []

        for edge_idx in self.connectedEdges:
            if (int(self.edgesList[edge_idx, 0]) != node_idx):
                neigh_nodes.append(int(self.edgesList[edge_idx, 0]))
            else:
                neigh_nodes.append(int(self.edgesList[edge_idx, 1]))

        return neigh_nodes

    def plotNetwork(self):
        '''Plot the pipeline network.'''

        for i, node in enumerate(self.nodesList):
            plt.scatter(node[0], node[1])
            plt.text(node[0] - 3, node[1] + 3, f'{i+1}', fontsize=8, ha='right')

            for edge_idx in self.connectedEdges[i]:
                edge = self.edgesList[edge_idx]
                nodeA_idx = int(edge[0])
                nodeB_idx = int(edge[1])
                nodeA = self.nodesList[nodeA_idx]
                nodeB = self.nodesList[nodeB_idx]
                plt.plot([nodeA[0], nodeB[0]], [nodeA[1], nodeB[1]])

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Pipeline Network")
        plt.legend()
        plt.show()