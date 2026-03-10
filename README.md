The idea behind this repository was inspired by the ICCAD 2025 contest, specifically on the problem regarding Trojan detection from gate-level netlist.

The folders in this repo contain a pipeline of getting RTL designs and determining what gates in the inserted designs are trojaned or not trojaned, using a Graph SAGE + GCN model. 

The SCOAP values used as features in this project were gotten from the "Testability Measurement Tool" seen in https://sourceforge.net/projects/testabilitymeasurementtool/. Additional extracted information such as the indegree and outdegree of each node, gate type, and distance to PIs & POs were also measured and used.

We are able to detect trojaned gates with a TPR of 93.58% and TNR of 98.1%, giving an average F1 Score of about 94.3%.

Contributors: Tamar Spira, Ayomide Adekoya
