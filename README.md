# Pipeline-Network-Design
Design a natural gas pipeline network for residential areas.

This is a project for the Applied Fluid Mechanics course in the fifth semester of the Mechanical Engineering department at Ntua.

The network in Figures/Figure 1 covers the natural gas needs of houses connected to residential areas' network nodes. Gas enters the network at node 1 and the total pressure of the network at node 1 is 30 mbar. The total volume flow rate of the network is 1000 m^3/h. The demand for the design volume flow rate by consumers (m3/h) is given in the Figure 1.

The task is to select suitable diameters for the pipes of the network. Standard diameters for polyethylene pipes SDR 11 from Figures/Table 1.

1) Network design requirements:
(a) The pressure should not drop below 25.5 mbar at any point in the network.
(b) The flow velocity should not exceed 8 m/s in any pipeline of the network for the avoidance of whistling.

2) Calculate the variation in operating pressure at the network nodes during the day. Values of daily flow variation are provided in Figures/Table 2.

The attributes of the pipelines, such as the initial diameter (to be updated) and roughness, as well as certain parameters related to the problem, such as the density and viscosity of natural gas, are specified in the "params.json" file. The "boundary_conditions_nodes" parameter is a list referring to the nodes that have boundary conditions.

Inside "network_data.xlsx", there are two sheets. In the "Nodes" sheet, I have specified the coordinates, initial pressure values, and consumption for each node. The initial pressures are random values (except for the first one, which serves as a boundary condition for our problem) and will be updated using the iterative Newton-Raphson algorithm to meet the requirements of the problem. Similarly, in the "Edges" sheet, I have specified the nodes of each pipeline and its length.
