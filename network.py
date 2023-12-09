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
        0: node A
        1: node B
        2: lenght
        3: diameter
        4: epsilon (roughness)
        5: dP (pressure difference)
        6: k: coefficient of linear losses
        7: Q (supply)
        '''

        self.connectedEdges = self.init_connectedEdges()  # the connected edges for each node are added to the list when createNetwork.edgeAttributes() function is called. This is done for time optimazation.
        '''Array with the connected edges for each node.'''
        self.edgeAttributes()  
        
    def init_connectedEdges(self) -> list:
        '''Create an empty list with number of rows equal to the number of nodes and a varying number of columns.'''

        init_con = []  
        
        for i in range(len(self.nodesList)):
            init_con.append([])

        return init_con

    def edgeAttributes(self):
        '''Calculate the attributes of the network's edges.'''

        self.edgesList[:, 4] = params["epsilon"]
        self.edgesList[:, :2] -= 1  # reduce the index of the nodes by 1 for better usability with array indexing

        for i, edge in enumerate(self.edgesList):
            edge[3] = params["D"]
            nodeA_idx = int(edge[0])  # data type casting beacuse the type of nodesList array is float
            nodeB_idx = int(edge[1])
            edge[5] = self.nodesList[nodeA_idx, 2] - self.nodesList[nodeB_idx, 2]  # dP = P_A - P_B 
            edge[6] = self.k_coefficient(edge[4], edge[3], edge[2])
            edge[7] = self.supply(edge[5], edge[6])

            # add the edge to the connectedEdges list of the respective nodes
            self.connectedEdges[nodeA_idx].append(i)
            self.connectedEdges[nodeB_idx].append(i)
    
    def k_coefficient(self, epsilon: float, D: float, L: float) -> float:
        '''Calculate the k coefficient of an edge.'''

        Re = 1.0e10  # Reynolds: initial value set to infinite
        # lambda_ = pow(1 / (1.14 - 2*log10(self.epsilon/self.D + 21.25/Re**0.9)), 2)
        lambda_ = pow(1 / (1.14 - 2*log10(epsilon/D)), 2)
        k = lambda_ * (L/D) * 8 / (pi**2 * 9.81 * D**4)

        return k
    
    # def k_coefficient(self) -> float:
    #     '''Calculate the k coefficient of the edge.'''

    #     Re = 1.0e10  # Reynolds: initial value set to infinite
    #     k_old = 5  # random initial value

    #     while True:
    #         lambda_ = pow(1 / (1.14 - 2*log10(self.epsilon/self.D + 21.25/Re**0.9)), 2)
    #         k = lambda_ * (self.L/self.D) * 8 / (pi**2 * 9.81 * self.D**4)
    #         Q = abs(self.supply(k))
    #         U = 4*Q / (pi*self.D**2)  # velosity
    #         Re = (U*self.D) / params["viscosity"]

    #         if (abs((k - k_old)/k) <= 1.0e-7):  # check if converged
    #             break

    #         k_old = k
    #         # print(str(i) + ": " + str(k_old))

    #     return k

    def supply(self, dP: float, k: float) -> float:
        '''Calculate the supply Q of an edge.'''

        return sqrt(abs(dP) / k)
    
    def sign_Q(self, edge_idx: int, ref_node_idx: int) -> float:
        '''
        Get the supply of an edge with the right sign given the reference node.
        
        Parameters
        ----------
        edge : index of the edge to calculate the supply
        ref_node : index of the reference node in the edge
        '''

        sign = 1
        if (ref_node_idx == self.edgesList[edge_idx, 0] and self.edgesList[edge_idx, 5] > 0):  # check if ref_node is equal with the nodeA
            sign = -1
        elif (ref_node_idx == self.edgesList[edge_idx, 1] and self.edgesList[edge_idx, 5] < 0):  # check if ref_node is equal with the nodeB
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