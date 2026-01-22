# Adaptive Neural Memory for LLMs

> **A PyTorch-based implementation of human-like, adaptive memory for Artificial Intelligence**, designed to *learn*, *forget*, and *evolve* without relying on conventional databases.

---

## Problem Statement

Modern Large Language Models (LLMs) suffer from a **static memory limitation**. While approaches like **RAG (Retrieval-Augmented Generation)** allow models to *retrieve* information, they behave more like a search engine than a human brain.

### Why Static Memory Fails

* **No Adaptation**
  Traditional databases store all data equally. They do not adapt to changing user preferences or long-term behavioral patterns.

* **Data Pollution**
  Without a forgetting mechanism, long interaction histories become noisy, causing irrelevant or outdated information to dominate retrieval.

* **Lack of Behavioral Nuance**
  Vector stores capture semantic similarity but miss *implicit signals* such as tone, engagement patterns, and preferred explanation styles.

---

## The Solution

This project introduces a **dynamic, tensor-based neural memory system** built in **PyTorch**.

Instead of storing raw text in databases (MySQL, MongoDB, Vector DBs), the system uses a **differentiable neural memory module** that:

* Learns from repetition
* Forgets unused information
* Evolves through reinforcement-style feedback

---

## Design Goals

The architecture is inspired by **biological memory systems** and applies them to deep learning:

1. **Adaptive Memory**
   Identifies important user traits (e.g., prefers diagrams vs. text) and reinforces them through repeated exposure.

2. **Forgetting Mechanism**
   Applies a decay function so unused memories fade over time, keeping the context clean and relevant.

3. **Personalization**
   Encodes behavioral signals such as time-of-use, tone, and content preferences into memory embeddings.

4. **Scalability (No Database)**
   Uses compressed PyTorch tensors instead of raw conversation logs, enabling efficient long-term memory handling.

---

## High-Level Architecture

The system follows a **left-to-right information flow**, clearly separating decision-making from storage.

```
![Conceptual_Design](https://github.com/user-attachments/assets/1b85bfc6-c8d9-4bdc-865d-264d276476e2)

```

### Core Components

* **Input Encoder**
  Converts user text and implicit signals (time, tone, interaction style) into dense embeddings.

* **Controller (RL Agent)**
  Decides whether information is worth storing and determines its memory destination.

* **Split Memory System**
  Dual-memory structure inspired by human cognition (STM & LTM).

* **Retrieval Attention**
  Fetches only the most relevant memories for prompt injection.

---

## Memory Strategy

This project goes beyond vector databases by implementing a **Hierarchical Memory Architecture**.

### 1️. Short-Term Memory (STM) vs Long-Term Memory (LTM)

* **STM (Short-Term Memory)**
  High-fidelity, volatile buffer storing recent interactions (active session context).

* **LTM (Long-Term Memory)**
  Persistent PyTorch tensor that stores recurring habits and stable preferences.

Only high-confidence patterns are promoted from STM → LTM.

---

### 2️. Write Gating (Memory Filter)

To prevent memory pollution, a **Sigmoid Gating Network** filters incoming signals.

```text
Importance Score = Sigmoid(Controller Output)
```

* Low-signal inputs (`"Hi"`, `"Okay"`) → ignored
* High-signal inputs (preferences, habits) → stored in LTM

---

### 3️. Memory Consolidation

The controller monitors STM for repeated patterns.

* Frequently observed behaviors are **consolidated** into LTM
* Existing memory slots are reinforced instead of duplicated

This mimics **human memory consolidation during learning**.

---

### 4️. Decay (Forgetting Curve)

Each LTM memory has an associated strength score.

```math
S_{t+1} = S_t × λ
```

* `λ < 1` controls forgetting speed
* Rarely accessed memories fall below a retrieval threshold

Result: **Automatic forgetting of stale information**.

---

## How the LLM Uses Memory

Instead of injecting full conversation history into the prompt, the system uses **attention-based retrieval**.

### Retrieval Pipeline

1. **Query Vector Generation**
   The current user query is encoded into a dense vector.

2. **Attention-Based Matching**
   Cosine similarity is computed against LTM vectors, weighted by memory strength.

3. **Top-k Context Injection**
   Only the most relevant memories are injected into the system prompt.

Example:

> Memory: *"User prefers visual explanations"*
> Result: LLM responds with diagrams or bullet points instead of long text.

---

## Getting Started

### Prerequisites

* Python 3.8+
* PyTorch
* NumPy

---

### Installation

```bash
pip install torch numpy
```

---

### Usage

Run the main prototype script:

```bash
python memory_system.py
```

---

### Expected Output

The simulation demonstrates how the system:

*  Learns user preferences (e.g., *"I like visuals"*)
*  Ignores noise (e.g., *"Hi"*)
*  Forgets unused memories via decay
*  Retrieves the correct preference when queried later

---

##  Key Takeaway

This project showcases how **neural memory can be adaptive, selective, and biologically inspired**, enabling LLMs to behave less like static databases and more like evolving intelligent agents.

---

 *If you're exploring long-term memory for LLMs, this architecture provides a strong foundation for research and production systems.*
