# Foundational Models for Structured Data

This repository is a one-stop shop and community-maintained resource for the emerging field of **Relational Deep Learning (RDL)**. It is based on the report, "The Relational Awakening: From Parallel Paths to Foundational Models for Structured Data."

This repo charts the evolution of deep learning on structured data, from the 2017 divergence of Transformers (for unstructured data) and Graph Neural Networks (GNNs) to their modern synthesis in **Relational Foundation Models**.

## 🚀 Quick Links

  * **The Benchmark:** [RelBench (Stanford SNAP)](https://relbench.stanford.edu/) | [GitHub](https://github.com/snap-stanford/relbench)
  * **Key Model Comparison:** [See the SOTA Architecture Analysis](https://www.google.com/search?q=%23-architectures-for-a-relational-foundation)
  * **Open Problems:** [See the Future Research Roadmap](https://www.google.com/search?q=%23-the-new-frontier-open-problems)

## Table of Contents

1.  [The 2017 Bifurcation: Parallel Paths](https://www.google.com/search?q=%23-i-the-2017-bifurcation-parallel-paths)
2.  [The GNN "Underground": Industrial Adoption](https://www.google.com/search?q=%23-ii-the-gnn-underground-industrial-adoption)
3.  [The Bridge: Fusing Attention and Structure](https://www.google.com/search?q=%23-iii-the-bridge-fusing-attention-and-structure)
4.  [The Relational Awakening: Formalizing RDL](https://www.google.com/search?q=%23-iv-the-relational-awakening-formalizing-rdl)
5.  [Architectures for a Relational Foundation](https://www.google.com/search?q=%23-v-architectures-for-a-relational-foundation)
6.  [Industrialization of RDL: Kumo.ai Case Study](https://www.google.com/search?q=%23-vi-the-industrialization-of-rdl-kumoai)
7.  [A Generative Detour: Diffusion Models](https://www.google.com/search?q=%23-vii-a-generative-detour-diffusion-models)
8.  [The New Frontier: Open Problems](https://www.google.com/search?q=%23-viii-the-new-frontier-open-problems)
9.  [Contributing](https://www.google.com/search?q=%23-contributing)
10. [Works Cited](https://www.google.com/search?q=%23-works-cited)

-----

## I. The 2017 Bifurcation: Parallel Paths

The year 2017 marked a pivotal divergence in deep learning, defined by two seminal papers submitted just five days apart.

| Domain | Paper | Key Idea |
| :--- | :--- | :--- |
| **Unstructured Data (Sequences)** | **"Attention is All You Need"** [1] | Introduced the **Transformer**, which uses a self-attention mechanism. It's highly parallelizable and captures long-range dependencies, igniting the LLM boom. |
| **Structured Data (Graphs)** | **"Inductive Representation Learning on Large Graphs"** [5] | Introduced **GraphSAGE**, an *inductive* framework that learns functions to aggregate neighborhood features. This allowed GNNs to generalize to unseen nodes and scale to massive industrial graphs. |

This split created two parallel research tracks: one for the unstructured public internet (LLMs) and one for structured, proprietary business data (GNNs).

## II. The GNN "Underground": Industrial Adoption

While LLMs captured public attention, GNNs were quietly operationalized for high-value enterprise tasks. This was driven by a new generation of open-source libraries.

### Key Enablers (Open-Source Libraries)

  * **PyTorch Geometric (PyG)** [11]: A library for deep learning on graphs, providing scalable mini-batch loaders, multi-GPU support, and first-class support for heterogeneous graphs.
  * **Deep Graph Library (DGL)** [34]: A framework-agnostic library (supporting PyTorch, TensorFlow, etc.) for easing deep learning on graphs.

### Industrial Applications

Armed with these tools, companies like **Amazon** [7] and **Spotify** [9] deployed GNNs at scale for:

  * **Fraud Detection**: Modeling financial transactions as heterogeneous graphs (e.g., nodes for transactions, devices, IPs) to perform node classification (fraudulent or not). [7]
  * **Recommendation Engines**: Treating users and items as a bipartite graph to learn rich embeddings for a user based on the items they've interacted with (and vice-versa).

This industrial use of heterogeneous GNNs was an early, ad-hoc form of Relational Deep Learning.

## III. The Bridge: Fusing Attention and Structure

Traditional message-passing GNNs hit a "glass ceiling" due to several key limitations:

  * **Over-smoothing**: As layers get deeper, node embeddings converge to a single, indistinct value. [41]
  * **Over-squashing**: Information from exponentially large neighborhoods is "squashed" into a fixed-size vector, creating a bottleneck. [28]
  * **Long-Range Dependencies**: Capturing information between distant nodes requires many layers, which triggers the problems above. [41]

The **Graph Transformer** emerged as the solution, a hybrid architecture that fuses the global reasoning of Transformers with the structural awareness of GNNs. Models like **Graphormer** [12] and **GraphGPS** [42] proved that a "stacking" of GNN layers (for local bias) and Transformer layers (for global reasoning) is the state-of-the-art.

### GNN vs. Graph Transformer: A Comparative Framework

| Aspect | Graph Neural Networks (GNNs) | Graph Transformers (GTs) |
| :--- | :--- | :--- |
| **Information Flow** | Sequential message passing between connected nodes [41] | Self-attention; direct aggregation from all nodes [41] |
| **Long-Range Dependencies** | Struggle; require many layers (hops) [41] | Captured effectively in a single layer [41] |
| **Over-Smoothing** | Prone with deep layers; embeddings become too similar [41] | Less prone; attention allows flexible information mixing [41] |
| **Over-Squashing** | Prone; information compressed when aggregating many neighbors [28] | Mitigated by attention, distributing information effectively [41] |
| **Scalability** | Efficient; typically linear in number of nodes + edges | Poor; standard attention is $O(N^2)$ in number of nodes [43] |
| **Inductive Bias** | Strong; hard-coded for locality and graph topology | Weak; assumes a fully connected graph by default |

## IV. The Relational Awakening: Formalizing RDL

The ad-hoc industrial work and new hybrid architectures coalesced into a formal field: **Relational Deep Learning (RDL)** [13], [14].

The RDL blueprint is a simple, powerful idea:

1.  **Nodes**: Create a node for each **row** in each table.
2.  **Edges**: Create edges based on **primary-foreign key links**.
3.  **Graph Type**: The result is inherently a **temporal, heterogeneous graph**.

This formalization highlighted a "generalization chasm": a GNN trained on one database schema couldn't be used on another. [26] This created the need for a standardized benchmark to build and test generalist models.

### The "ImageNet" for RDL: RelBench

The **Relational Deep Learning Benchmark (RelBench)** [15] was developed by Stanford to fill this need. It is the foundational infrastructure for RDL research, providing:

  * **Realistic Datasets**: Large-scale, diverse datasets from e-commerce, social networks, etc.
  * **Standardized Tasks**: 30 challenging, domain-specific predictive tasks.
  * **Unified Pipelines**: Standard data splits, evaluation pipelines, and baselines.

**Links:**

  * **Website:** [relbench.stanford.edu](https://relbench.stanford.edu/)
  * **GitHub:** [github.com/snap-stanford/relbench](https://github.com/snap-stanford/relbench)

RDL models trained on RelBench were shown to **reduce model development time by over 95%** by automating the time-consuming process of manual feature engineering. [46], [49]

## V. Architectures for a Relational Foundation

The creation of RelBench spurred a new wave of SOTA architectures. Here is a comparative analysis of the key models.

### Comparative Analysis of Modern Relational Models

| Model | Core Architecture | Key Innovation | Handling of RDBs | Open Source |
| :--- | :--- | :--- | :--- | :--- |
| **HGT+PE** [17], [52] | Heterogeneous Transformer + Laplacian PE | Baseline GT for heterogeneous graphs. | Converts RDB to graph; uses expensive, general PE for structure. [54] | **Yes** [55], [56] |
| **RelGT** [18] | Hybrid (Local+Global) Graph Transformer | **Multi-element tokenization** (type, hop, time, structure) instead of one expensive PE. | Converts RDB to graph; time is a feature. | **Yes** [18] |
| **RGP** [19] | Perceiver-based Graph Transformer | **Active temporal sampler** + struct/temporal fusion. Treats time as an active sampling dimension. | Converts RDB to graph; time is an active sampling dimension. | Not Specified |
| **Griffin** [20] | MPNN (GNN) + Cross-Attention | **Foundation Model Pretraining** (masked value completion) for transfer learning. | Converts RDB to graph; pretrains for transferability. | **Yes** [67] |
| **DBFormer** [21] | Relational-native Transformer | **"Graph-less" approach**. Message passing based on the "formal relational model." | **Does not convert to graph**; models relations directly. | **Yes** [21] |

This analysis reveals a fundamental fork in the road:

  * **Graph-Centric Path (Griffin, RelGT)**: Convert the database *to* a graph, then apply a Graph Transformer.
  * **Relational-Centric Path (DBFormer)**: Build a new Transformer that *natively* understands relational algebra.

## VI. The Industrialization of RDL: Kumo.ai

The **Kumo.ai** platform [22] is a case study in how RDL concepts are being productized. It provides a SaaS "easy button" that connects to a data warehouse (e.g., Snowflake) and automates the entire RDL pipeline. [78]

Kumo's core technology, the KumoRFM [23], is not a single model but a hybrid **"mixture-of-experts"** platform. It maintains a toolbox of SOTA architectures and selects the right one for the task:

  * **Classic GNNs**: Uses **GraphSAGE** for cold-start problems and **GIN** for frequency signals. [83]
  * **Hybrid GNNs**: Deploys custom hybrids for tasks like recommendations. [84]
  * **Relational Graph Transformers (RGTs)**: Uses RGTs with time embeddings to handle complex, temporal data. [85]
  * **LLM Integration**: Integrates GNNs with LLMs (like Snowflake Cortex) to incorporate unstructured data (e.g., product descriptions). [82]

Kumo.ai demonstrates that the industrial "foundation model" is a practical, hybrid AutoML platform that aggregates the entire RDL research field.

## VII. A Generative Detour: Diffusion Models

Can diffusion models be used for predictions on relational data?
**The answer is primarily *indirectly***. Diffusion models are *generative* [90] and are used to support downstream *predictive* models (like GNNs or XGBoost).

Their two main uses are: [24]

1.  **Data Augmentation**: Models like **TabDDPM** [89] generate new, synthetic tabular data. This is most often used to **address class imbalance** (e.g., generating more "fraud" samples for a classifier). [25]
2.  **Data Imputation**: Models like **CSDI** fill in missing or incomplete entries [24], allowing predictive models to be trained on cleaner data.

This is now being extended to multi-table data with models like **RelDiff** [93], which generates entire synthetic relational databases.

### Analysis of Diffusion Models for Tabular/Relational Data

| Model | Primary Task | Use Case | Open Source |
| :--- | :--- | :--- | :--- |
| **TabDDPM** [89] | Generative | Data Augmentation (for prediction) | **Yes** |
| **TabDiff** [91] | Generative | Data Augmentation, Imputation | **Yes** |
| **RelDiff** [93] | Generative | Relational Data Synthesis (for prediction) | **Yes** |
| **DiffRI** [94] | Predictive | Relational Inference (Time Series) | Not Specified |

## VIII. The New Frontier: Open Problems

The convergence of structured data, GNNs, and Transformers has opened a new frontier. The field is moving from task-specific models to general-purpose foundation models. The following roadmap outlines the key open challenges and opportunities.

### Future Opportunities and Open Problems in Relational AI

| Opportunity / Open Problem | Domain | Difficulty | Description & Rationale |
| :--- | :--- | :--- | :--- |
| **1. GNN Architectural Limits** | Architecture | Medium | **Problem**: Over-smoothing [41] and over-squashing [28] remain fundamental bottlenecks in the GNN components of hybrid models. **Opportunity**: Develop more expressive GNN aggregators that can capture long-range info without these bottlenecks. [27] |
| **2. "Data-Centric AI" for RDL** | Data-Centric AI | Medium | **Problem**: The "RDB-to-graph" conversion is a critical, high-variance design choice, not just pre-processing. [77] **Opportunity**: Develop a framework for "automated graph schema engineering" that learns the optimal graph representation for a given task. |
| **3. True, End-to-End Multi-Modal RDL** | Architecture | High | **Problem**: Real RDBs are multi-modal (text, images, numbers). Current SOTA (e.g., Kumo [82]) bolts on an LLM (late-fusion). **Opportunity**: A single, unified model that can natively attend to graph structure, sequential text, and grid-like images. [101] |
| **4. The Neuro-Symbolic Database** | Neuro-Symbolic AI | High / "Holy Grail" | **Problem**: A fork exists between sub-symbolic (Griffin [20]) and symbolic (DBFormer [21]) approaches. **Opportunity**: Unify them. A "neural query processor" that can jointly learn to write an optimal SQL query (symbolic) and perform the predictive reasoning (sub-symbolic). |
| **5. True "Zero-Shot" RDB Generalization** | Foundation Models | High | **Problem**: The "Generalization Chasm." [26] How do you pretrain a model to generalize to a completely unseen database schema? **Opportunity**: Develop novel pretraining tasks (beyond Griffin's [69]) that teach a model abstract relational concepts (e.g., "many-to-many") independent of column names. |
| **6. Trust, Explainability, and Robustness** | AI Safety / XAI | Medium | **Problem**: Evergreen challenges. How do you explain a prediction from 5 tables and 10 hops? How do you guarantee privacy? [98] **Opportunity**: Adapt diffusion models for privacy-preserving synthetic RDBs [24]. Develop GNN explainability methods (e.g., GNNExplainer) for RDL's temporal, heterogeneous structures. [103] |

## 🤝 Contributing

This is a living document and a community resource. Contributions are welcome\! If you see a new paper, model, or resource that should be included, please open an **Issue** or submit a **Pull Request**.

-----

## Works Cited

[1] [1706.03762] Attention Is All You Need - arXiv
[2] Attention is All you Need - NIPS papers
[3] Attention is All you Need - NIPS papers (PDF)
[4] GraphSAGE: Inductive Representation Learning on Large Graphs - SNAP: Stanford
[5] [1706.02216] Inductive Representation Learning on Large Graphs - arXiv
[6] williamleif/GraphSAGE - GitHub
[7] Build a GNN-based real-time fraud detection solution - Amazon AWS
[8] How AWS uses graph neural networks - Amazon Science
[9] Analysing Privacy Policies... case studies of Tinder and Spotify - NIH
[10] PyTorch Geometric - Read the Docs
[11] pyg-team/pytorch\_geometric - GitHub
[12] Do Transformers Really Perform Badly for Graph Representation? - NeurIPS
[13] [2312.04615] Relational Deep Learning: Graph Representation Learning on Relational Databases - arXiv
[14] ICML Poster Position: Relational Deep Learning - ICML 2025
[15] RelBench: Relational Deep Learning Benchmark - RelBench
[16] snap-stanford/relbench - GitHub
[17] [2003.01332] Heterogeneous Graph Transformer - arXiv
[18] [2505.10960] Relational Graph Transformer - arXiv
[19] Integrating Temporal and Structural Context in Graph Transformers for RDL - arXiv
[20] Griffin: Towards a Graph-Centric Relational Database Foundation Model - arXiv
[21] Transformers Meet Relational Databases - arXiv
[22] Kumo: The Ultimate Guide to Predictive AI on Relational Data - Skywork
[23] KumoRFM: A Foundation Model for In-Context Learning on Relational Data - Kumo.ai
[24] awesome-diffusion-models-for-tabular-data - GitHub
[25] (PDF) RelDiff: Relational Data Generative Modeling... - ResearchGate
[26] Graph foundation models for relational data - Google Research
[27] Open Problems in Graph Neural Networks - Medium
[28] On the Bottleneck of Graph Neural Networks - OpenReview
[29] How Transformers Work - DataCamp
[30] “Attention is all you need” in plain English - Medium
[31] Attention Is All You Need — And All We’ve Lost - Medium
[32] LLM Transformer Model Visually Explained - Polo Club of Data Science
[33] Deep Graph Library
[34] dmlc/dgl - GitHub
[35] torch\_geometric.datasets - PyTorch Geometric
[36] PyTorch Geometric vs Deep Graph Library - Medium
[37] 7 Open Source Libraries for Deep Learning Graphs - Exxact Corporation
[38] Build a GNN-based real-time fraud detection solution using... - Amazon AWS
[39] [D] Are GNNs obsolete because of transformers? - Reddit
[40] Comparative Analysis of GNNs and Transformers for Fake News Detection - MDPI
[41] An Introduction to Graph Transformers - Kumo.ai
[42] Graph Transformer — pytorch\_geometric documentation
[43] Graph Transformer - Medium
[44] The Relational Awakening... (ContextGNN → RELGT) - Medium
[45] Self-Service ML with Relational Deep Learning - Medium
[46] Relational Deep Learning: Challenges, Foundations... - arXiv
[47] [2407.20060] RelBench: A Benchmark for Deep Learning on Relational Databases - arXiv
[48] RelGNN: Composite Message Passing for RDL - GitHub
[49] RelBench: A Benchmark for Deep Learning on Relational Database - Kumo AI
[50] Heterogeneous Graph Generation... - arXiv
[51] HHGT: Hierarchical Heterogeneous Graph Transformer - arXiv
[52] Relational Graph Transformer - arXiv (v1)
[53] Relative performance drop (%) when position encoding (PE) is removed... - ResearchGate
[54] Runtime Comparison of HGT and HGT+PE... - ResearchGate
[55] acbull/HGT-DGL - GitHub
[56] acbull/pyHGT - GitHub
[57] [2511.04557] Integrating Temporal and Structural Context... - arXiv
[58] RGP: A Cross-Attention based Graph Transformer... - OpenReview
[59] Integrating Temporal and Structural Context... (PDF) - OpenReview
[60] [Papierüberprüfung] Integrating Temporal and Structural Context... - Moonlight
[61] [Literature Review] Integrating Temporal and Structural Context... - Moonlight
[62] (PDF) Integrating Temporal and Structural Context... - ResearchGate
[63] Rishit-dagli/Perceiver - GitHub
[64] krasserm/perceiver-io - GitHub
[65] lucidrains/perceiver-pytorch - GitHub
[66] Google's Griffin Architecture... - YouTube
[67] Griffin: Towards a Graph-Centric... - OpenReview
[68] Griffin: Towards a Graph-Centric... - arXiv (v2)
[69] Griffin: Towards a Graph-Centric... (PDF) - Amazon Science
[70] Griffin: Towards a Graph-Centric... - ResearchGate
[71] yanxwb/Griffin - GitHub
[72] DBFormer: A Dual-Branch Adaptive Remote Sensing... - ResearchGate
[73] DBFormer: A Dual-Branch Adaptive... - Semantic Scholar
[74] DBFormer: A Dual-Branch Adaptive Remote Sensing... - MDPI
[75] Transformers Meet Relational Databases - arXiv (HTML)
[76] Transformers Meet Relational Databases - ChatPaper
[77] A Data-Centric AI Paradigm... - MDPI
[78] Kumo: AI models for relational data
[79] Kumo AI SaaS Security White Paper
[80] Architecture and Security - Kumo Documentation
[81] Graph Neural Networks (GNNs): Introduction and examples - Kumo.ai
[82] Enabling Kumo's predictive AI natively in Snowflake
[83] What model architectures does Kumo incorporate... - Kumo Docs
[84] Hybrid Graph Neural Networks - Kumo.ai
[85] Relational Graph Transformers - Kumo.ai
[86] Growth and Marketing - Kumo.ai
[87] Improving recommendation systems with LLMs and Graph Transformers - Kumo.ai
[88] TabDDPM: Modelling Tabular Data with Diffusion Models - arXiv
[89] TabDDPM: Modelling Tabular Data with Diffusion Models - PMLR
[90] When to Use GenAI Versus Predictive AI - MIT Sloan
[91] TabDiff: a Mixed-type Diffusion Model... - OpenReview
[92] Relational Data Generation with GNNs and Latent Diffusion - OpenReview
[93] RelDiff: Relational Data Generative Modeling... - arXiv
[94] Diffusion Model for Relational Inference - arXiv
[95] [2305.15321] Towards Foundation Models for Relational Databases [Vision Paper] - arXiv
[96] Challenges and Trade-Offs for Graph Neural Networks - Oregon State
[97] Relational Programming with Foundation Models - arXiv
[98] On the Opportunities and Risks of Foundation Models - Stanford CRFM
[99] Researchers at Stanford Present RelBench... - Reddit
[100] Data-Centric AI vs. Model-Centric AI - MIT
[101] PyG 2.0: Scalable Learning on Real World Graphs - arXiv
[102] Graph Neural Networks for Databases: A Survey - arXiv
[103] Graph Machine Learning Explainability with PyG - Medium
