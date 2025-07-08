The idea behind this repository was inspired by the ICCAD 2025 contest, specifically on the problem regarding Trojan detection from gate-level netlist.

The folders in this repo contain a pipeline of getting RTL designs and determining what gates in the inserted designs are trojaned or not trojaned, using SCOAP values gotten from the "Testability Measurement Tool" seen in https://sourceforge.net/projects/testabilitymeasurementtool/.

Using these scores and two machine learning techniques (k-means clustering & GNN), we are able to detect trojaned gates with a TPR of _____ and TNR of ______.

Contributors: Tamar Spira, Ayomide Adekoya