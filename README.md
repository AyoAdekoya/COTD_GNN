**HARDWARE TROJAN GATE DETECTION FROM GATE-LEVEL NETLISTS**

This repository presents a graph-based machine learning pipeline for detecting Trojan-inserted gates in digital circuits starting from gate-level netlists. The project was inspired by the ICCAD 2025 Contest, specifically the challenge focused on Trojan detection from gate-level netlists. To build and evaluate the pipeline, we used benchmark designs from both:
1. ICCAD 2025 Contest benchmarks
2. Trust-Hub chip-level Trojan benchmarks: https://www.trust-hub.org/#/benchmarks/chip-level-trojan

**Project Overview**

  The goal of this repository is to take circuit designs and predict which **gates in a Trojan-inserted design are malicious**. To do this, we convert each gate-level netlist into a graph and apply a GraphSAGE + GCN model for node-level classification. Each node in the graph represents a gate, and the model predicts whether that gate is Trojaned or not.
  This approach helps reduce the time and cost of hardware Trojan analysis by giving an IC test engineer a more localized region of the chip to inspect, rather than requiring a full manual search across the entire design.

**Graph/Circuit Features**
1. The following SCOAP-based features were used: CC0, CC1, CO. These values were generated using the Testability Measurement Tool: https://sourceforge.net/projects/testabilitymeasurementtool/
2. Gate type
3. Indegree
4. Outdegree
5. Distance to primary inputs (PIs)
6. Distance to primary outputs (POs)

**Model**

The repository uses a hybrid GraphSAGE + GCN architecture for Trojan gate detection. At a high level, the pipeline:
1. Parses gate-level netlists
2. Converts them into graph representations
3. Extracts node-level features
4. Trains a graph neural network to classify each gate as Trojaned or non-Trojaned

Our Results:
1. ICCAD Benchmarks only: TPR = 90.21%, TNR = 98.19%, F1 = 92.02%
2. TrustHub Benchmarks only: TPR = 86.58%, TNR = 99.78%, F1 = 78.72%
3. ICCAD & TrustHub Benchmarks: TPR = 95.38%, TNR = 99.97%, F1 = 95.57%


Contributors: Tamar Spira, Ayomide Adekoya
