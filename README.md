# Pipeline-Network-Design
Design a natural gas pipeline network for residential areas.

This is a project for the Applied Fluid Mechanics course in the fifth semester of the Mechanical Engineering department at Ntua.

In this computational problem, the low-pressure natural gas network in Figures/Figure 1 is calculated. The network covers the natural gas needs of residential buildings connected to the network nodes. In the first part of the problem, appropriate diameters for the network pipes are selected, meeting specific design requirements. Standard diameters for polyethylene pipes SDR 11 are provided in Figures/Table 1. In the second part, the variation of operating pressure at the network nodes is calculated and presented throughout the day. Values of daily flow variation are provided in Figures/Table 2. The network is solved using the Newton-Raphson method and implemented in the Python programming language.

Gas enters the network at node 1 and the total pressure of the network at node 1 is 30 mbar. The total volume flow rate of the network is 1000 m^3/h. The demand for the design volume flow rate by consumers (m3/h) is given in the Figure 1. 

Network design requirements:
(a) The pressure should not drop below 25.5 mbar at any point in the network.
(b) The flow velocity should not exceed 8 m/s in any pipeline of the network for the avoidance of whistling.

The network solver program is written in Python and is divided into two files, `network.py` and `main.py`. In the `network.py` file, the construction of the network takes place, and any necessary data is calculated, such as pipe flows or the k coefficients. In the `main.py` file, all the steps for selecting the diameters developed earlier are implemented, such as solving the system with Newton-Raphson, selecting the k coefficients, and choosing the diameters. The program reads from an Excel file, `network_data.xlsx`, where all the data defining the topology and various parameters of the problem are stored. Specifically, in the Nodes sheet, the coordinates, initial pressures, and consumptions of the nodes are defined. In the Edges sheet, the nodes defining each pipe, their lengths, localized losses ζ (referring to the first node of each conduit), and standardized diameters are entered. The last two columns are filled according to the instructions displayed in the terminal when `main.py` is run. In the Params sheet, the roughness, kinematic viscosity, and density of natural gas are defined. In the column `nodes_with_boundary_conditions`, the nodes with boundary conditions are entered on each line. In the Daily demand sheet, the values of daily flow variation are defined. In the Standardized diameters sheet, the standardized diameters used are listed. The program can solve any pipeline network.
