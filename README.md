Project B — Cybersecurity: “Sequence-Level Intrusion Detection using Packet-Level 1D-CNN + RNN/LSTM Autoencoder”


Goal: Detect anomalous or malicious network sessions by modeling sequences of packets/flows with a hybrid CNN+RNN model for representation learning and anomaly scoring.


Dataset: UNSW-NB15 Dataset on Kaggle

Description:

Contains 49 features per network flow, including packet count, byte rate, flags, and protocol information.
Traffic includes normal and nine attack categories (Fuzzers, DoS, Exploits, Reconnaissance, Shellcode, etc.).

Goal:

The main goal of this project is to build an intelligent intrusion detection system that can automatically recognize abnormal or malicious network activity by analyzing sequential patterns of network flows.

Using the UNSW-NB15 dataset, which contains realistic modern network traffic and multiple attack types, students will:

Learn how to preprocess flow-based features and group them into meaningful sequences.
Apply 1D Convolutional Neural Networks (CNNs) to extract spatial correlations among packet or flow features.
Use LSTM Autoencoders to capture temporal dependencies and reconstruct normal network behavior over time.
Detects intrusions by identifying sequences that show high reconstruction error, indicating abnormal or attack patterns.
