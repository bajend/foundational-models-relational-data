# The Relational Awakening: Foundational Models for Structured Data

This repository is a one-stop shop and community-maintained resource for the emerging field of **Relational Deep Learning (RDL)**. 

This repo charts the evolution of deep learning on structured data, from the 2017 divergence of Transformers (for unstructured data) and Graph Neural Networks (GNNs) to their modern synthesis in **Relational Foundation Models**.

## 🚀 Quick Links

* **The Benchmark:** [RelBench (Stanford SNAP)][14] | [GitHub][15]
* **Key Model Comparison:** [See the SOTA Architecture Analysis](#v-architectures-for-a-relational-foundation)
* **Open Problems:** [See the Future Research Roadmap](#viii-the-new-frontier-open-problems)

## Table of Contents

1.  [The 2017 Bifurcation: Parallel Paths](#i-the-2017-bifurcation-parallel-paths)
2.  [The GNN "Underground": Industrial Adoption](#ii-the-gnn-underground-industrial-adoption)
3.  [The Bridge: Fusing Attention and Structure](#iii-the-bridge-fusing-attention-and-structure)
4.  [The Relational Awakening: Formalizing RDL](#iv-the-relational-awakening-formalizing-rdl)
5.  [Architectures for a Relational Foundation](#v-architectures-for-a-relational-foundation)
6.  [Industrialization of RDL: Kumo.ai Case Study](#vi-the-industrialization-of-rdl-kumoai)
7.  [A Generative Detour: Diffusion Models](#vii-a-generative-detour-diffusion-models)
8.  [The New Frontier: Open Problems](#viii-the-new-frontier-open-problems)
9.  [Contributing](#-contributing)
10. [Works Cited](#works-cited)

---

## I. The 2017 Bifurcation: Parallel Paths

The year 2017 marked a pivotal divergence in deep learning, defined by two seminal papers submitted just five days apart.



| Domain | Paper | Key Idea |
| :--- | :--- | :--- |
| **Unstructured Data (Sequences)** | **"Attention is All You Need"** [1] | Introduced the **Transformer**, which uses a self-attention mechanism. It's highly parallelizable and captures long-range dependencies [48], igniting the LLM boom [49], [51]. |
| **Structured Data (Graphs)** | **"Inductive Representation Learning on Large Graphs"** [2] | Introduced **GraphSAGE**, an *inductive* framework that learns functions to aggregate neighborhood features. This allowed GNNs to generalize to unseen nodes and scale to massive industrial graphs. [Code: [3]] |

This split created two parallel research tracks: one for the unstructured public internet (LLMs) and one for structured, proprietary business data (GNNs).

---

## II. The GNN "Underground": Industrial Adoption

While LLMs captured public attention, GNNs were quietly operationalized for high-value enterprise tasks. This was driven by a new generation of open-source libraries.

### Key Enablers (Open-Source Libraries)

* **PyTorch Geometric (PyG)** [8], [9]: A library for deep learning on graphs, providing scalable mini-batch loaders, multi-GPU support, and first-class support for heterogeneous graphs.
* **Deep Graph Library (DGL)** [52], [53]: A framework-agnostic library (supporting PyTorch, TensorFlow, etc.) for easing deep learning on graphs.

### Industrial Applications

Armed with these tools, companies like **Amazon** [4], [6] and **Spotify** [7] deployed GNNs at scale for:
* **Fraud Detection**: Modeling financial transactions as heterogeneous graphs (e.g., nodes for transactions, devices, IPs) to perform node classification (fraudulent or not). [4], [5]
* **Recommendation Engines**: Treating users and items as a bipartite graph to learn rich embeddings for a user based on the items they've interacted with (and vice-versa).

This industrial use of heterogeneous GNNs was an early, ad-hoc form of Relational Deep Learning.

---

## III. The Bridge: Fusing Attention and Structure

Traditional message-passing GNNs hit a "glass ceiling" due to several key limitations:
* **Over-smoothing**: As layers get deeper, node embeddings converge to a single, indistinct value. [58]
* **Over-squashing**: Information from exponentially large neighborhoods is "squashed" into a fixed-size vector, creating a bottleneck. [47]
* **Long-Range Dependencies**: Capturing information between distant nodes requires many layers, which triggers the problems above. [58]

The **Graph Transformer** emerged as the solution, a hybrid architecture that fuses the global reasoning of Transformers with the structural awareness of GNNs. Models like **Graphormer** [10] and **GraphGPS** [59] proved that a "stacking" of GNN layers (for local bias) and Transformer layers (for global reasoning) is the state-of-the-art.

---

## IV. The Relational Awakening: Formalizing RDL

The ad-hoc industrial work and new hybrid architectures coalesced into a formal field: **Relational Deep Learning (RDL)** [11], [12], [13].

The RDL blueprint is a simple, powerful idea:
1.  **Nodes**: Create a node for each **row** in each table.
2.  **Edges**: Create edges based on **primary-foreign key links**.
3.  **Graph Type**: The result is inherently a **temporal, heterogeneous graph** [13], [61], [62].



This formalization highlighted a "generalization chasm": a GNN trained on one database schema couldn't be used on another. [43] This created the need for a standardized benchmark to build and test generalist models.

### The "ImageNet" for RDL: RelBench

The **Relational Deep Learning Benchmark (RelBench)** [14] was developed by Stanford to fill this need. It is the foundational infrastructure for RDL research, providing:
* **Realistic Datasets**: Large-scale, diverse datasets from e-commerce, social networks, etc.
* **Standardized Tasks**: 30 challenging, domain-specific predictive tasks.
* **Unified Pipelines**: Standard data splits, evaluation pipelines, and baselines [15], [16].

**Links:**
* **Website:** [relbench.stanford.edu][14]
* **GitHub:** [github.com/snap-stanford/relbench][15]

RDL models trained on RelBench were shown to **reduce model development time by over 95%** by automating the time-consuming process of manual feature engineering. [13], [16]

---

## V. Architectures for a Relational Foundation

The creation of RelBench spurred a new wave of SOTA architectures. Here is a comparative analysis of the key models.

### Comparative Analysis of Modern Relational Models

| Model | Core Architecture | Key Innovation | Handling of RDBs | Open Source |
| :--- | :--- | :--- | :--- | :--- |
| **HGT+PE** [17], [20] | Heterogeneous Transformer + Laplacian PE | Baseline GT for heterogeneous graphs. | Converts RDB to graph; uses expensive, general PE for structure. [65], [66] | **Yes** [18], [19] |
| **RelGT** [20] | Hybrid (Local+Global) Graph Transformer | **Multi-element tokenization** (type, hop, time, structure) instead of one expensive PE. | Converts RDB to graph; time is a feature. | **Yes** [20] |
| **Relational Transformer (RT)** [21] | Set-Transformer Encoder | Tokenizes an **entire row** into one embedding. Uses learnable **'join embeddings'** to model PK-FK links. | Converts RDB to a **set of row embeddings** (not a graph). | **Yes** [22] |
| **RGP** [23] | Perceiver-based Graph Transformer | **Active temporal sampler** + struct/temporal fusion. Treats time as an active sampling dimension. | Converts RDB to graph; time is an active sampling dimension. | Not Specified |
| **Griffin** [24] | MPNN (GNN) + Cross-Attention | **Foundation Model Pretraining** (masked value completion) for transfer learning. | Converts RDB to graph; pretrains for transferability. | **Yes** [25] |
| **DBFormer** [26] | Relational-native Transformer | **"Graph-less" approach**. Message passing based on the "formal relational model." | **Does not convert to graph**; models relations directly. | **Yes** [26] |

---

### 5.1 The Evolutionary Path: How We Got Here

This timeline shows how each new architecture was built to solve the problems of the last.



1.  **Baseline: GNNs (e.g., GraphSAGE, GCN)**
    * **Problem:** Manual SQL feature engineering is slow, requires domain expertise, and is often suboptimal.
    * **Solution:** Convert the database into a heterogeneous graph and use a GNN to learn feature representations automatically.
    * **Impact:** This proved the RDL concept. It showed that GNNs "consistently outperformed traditional tabular models" [16] and could reduce development time by over 95% [13].

2.  **Hybrid: Graph Transformers (e.g., HGT+PE)**
    * **Problem:** GNNs' local message-passing architecture suffers from **over-smoothing** and **over-squashing** [58, 47]. They struggle to capture long-range, global relationships.
    * **Solution:** Augment the GNN with a Transformer, using self-attention to act as a "fully-connected" layer that can reason globally.
    * **Impact:** This became the default SOTA architecture. However, the baseline HGT [17] was a *general* graph model, and its positional encoding (PE) was computationally expensive and not designed for RDBs [65, 66].

3.  **Relational-Native: RelGT (Relational Graph Transformer)**
    * **Problem:** HGT+PE was slow, and its generic PE didn't efficiently capture the specific semantics of a database (time, tables, keys).
    * **Solution:** **RelGT** [20] created a *relational-native* tokenization. It replaced the single, expensive PE with a set of cheap, specific embeddings for node type (table), hop distance, and relative time.
    * **Impact:** This was a huge breakthrough. It was much faster and more accurate, proving that *RDB-specific* architectures are superior to *general-graph* models. It became the new champion on RelBench.

4.  **"Graph-less" Alternatives (RT & DBFormer)**
    * **Problem:** Is converting the RDB into a graph the right idea in the first place? This "graph-centric" view is complex and may lose information.
    * **Solution (RT):** The **Relational Transformer (RT)** [21] treats the database as a *set of rows* and uses "join embeddings" (based on PK-FK links) to model relations, avoiding GNNs entirely.
    * **Solution (DBFormer):** **DBFormer** [26] goes even further, creating a neural message-passing scheme that *natively follows the rules of relational algebra*.
    * **Impact:** These models challenge the entire "graph-centric" pipeline. They open a new and powerful "relational-centric" or neuro-symbolic research path.

5.  **The Specialists: RGP (Relational Graph Perceiver)**
    * **Problem:** Models like RelGT treat time as just another feature. They don't actively model *temporal context* (e.g., "what similar thing happened 6 months ago?").
    * **Solution:** **RGP** [23] uses a Perceiver architecture with an *active temporal sampler* that can look back and find distant-but-relevant events, fusing them with the local structural information.
    * **Impact:** As of November 2025, RGP is the new SOTA on RelBench, proving that specialized temporal reasoning is a key for high performance on enterprise data.

6.  **The Generalists: Griffin (The Foundation Model)**
    * **Problem:** All the models above are trained *from scratch* for *one task* on *one database*. They can't generalize (the "generalization chasm" [43]).
    * **Solution:** **Griffin** [24] is the first model designed as a true *foundation model*. It's pre-trained on a "masked value completion" task across many tables and DBs.
    * **Impact:** This is the first model that can be fine-tuned for new tasks, especially in low-data scenarios. It represents the entire field's shift from task-specific models to true, general-purpose relational intelligence.

---

### 5.2 What is the State-of-the-Art (Nov 2025)?

Based on the latest papers, the "state-of-the-art" (SOTA) is fragmented because different models are now solving for different *goals*:

* **For Pure Benchmark Performance (RelBench):** **RGP (Relational Graph Perceiver)** [23] is the new SOTA. Its paper (released November 2025) is the most recent and claims to "consistently outperform RelGT" by better-modeling time.
* **For Generalization & Transfer Learning:** **Griffin** [24] is the SOTA. It's the first true *foundation model* designed to be pre-trained and then fine-tuned for new tasks, which no other model is designed for.
* **For "Non-Graph" Neuro-Symbolic Approaches:** **DBFormer** [26] is the SOTA. It represents a completely different, "graph-less" path that models relational algebra directly.
* **For "Set-Based" (Non-Graph) Approaches:** The **Relational Transformer (RT)** [21] is the SOTA for this paradigm, which avoids GNNs entirely and instead models relations using "join embeddings."

In short, while **RGP** holds the top benchmark score, the most exciting "frontier" is the split between **Griffin** (the foundation model) and the **DBFormer** / **RT** (the non-graph-centric) models.

---

## VI. The Industrialization of RDL: Kumo.ai

The **Kumo.ai** platform [27], [29] is a case study in how RDL concepts are being productized. It provides a SaaS "easy button" that connects to a data warehouse (e.g., Snowflake) and automates the entire RDL pipeline. [30], [31]

Kumo's core technology, the KumoRFM [28], is not a single model but a hybrid **"mixture-of-experts"** platform. It maintains a toolbox of SOTA architectures and selects the right one for the task:
* **Classic GNNs**: Uses **GraphSAGE** for cold-start problems and **GIN** for frequency signals. [32], [34]
* **Hybrid GNNs**: Deploys custom hybrids for tasks like recommendations. [35]
* **Relational Graph Transformers (RGTs)**: Uses RGTs with time embeddings to handle complex, temporal data. [36]
* **LLM Integration**: Integrates GNNs with LLMs (like Snowflake Cortex) to incorporate unstructured data (e.g., product descriptions). [33], [38]

Kumo.ai demonstrates that the industrial "foundation model" is a practical, hybrid AutoML platform that aggregates the entire RDL research field.

---

## VII. A Generative Detour: Diffusion Models

Can diffusion models be used for predictions on relational data?
**The answer is primarily *indirectly***. Diffusion models are *generative* [76] and are used to support downstream *predictive* models (like GNNs or XGBoost).

Their two main uses are: [39]
1.  **Data Augmentation**: Models like **TabDDPM** [74], [75] generate new, synthetic tabular data. This is most often used to **address class imbalance** (e.g., generating more "fraud" samples for a classifier). [40]
2.  **Data Imputation**: Models like **CSDI** fill in missing or incomplete entries [39], allowing predictive models to be trained on cleaner data.

This is now being extended to multi-table data with models like **RelDiff** [40], [41], [42], which generates entire synthetic relational databases.

### Analysis of Diffusion Models for Tabular/Relational Data

| Model | Primary Task | Use Case | Open Source |
| :--- | :--- | :--- | :--- |
| **TabDDPM** [74] | Generative | Data Augmentation (for prediction) | **Yes** |
| **TabDiff** [77] | Generative | Data Augmentation, Imputation | **Yes** |
| **RelDiff** [42] | Generative | Relational Data Synthesis (for prediction) | **Yes** |
| **DiffRI** [78] | Predictive | Relational Inference (Time Series) | Not Specified |

---

## VIII. The New Frontier: Open Problems

The convergence of structured data, GNNs, and Transformers has opened a new frontier. The field is moving from task-specific models to general-purpose foundation models. The following roadmap outlines the key open challenges and opportunities.

### Future Opportunities and Open Problems in Relational AI

| Opportunity / Open Problem | Domain | Difficulty | Description & Rationale |
| :--- | :--- | :--- | :--- |
| **1. GNN Architectural Limits** | Architecture | Medium | **Problem**: Over-smoothing [58] and over-squashing [47] remain fundamental bottlenecks in the GNN components of hybrid models. **Opportunity**: Develop more expressive GNN aggregators that can capture long-range info without these bottlenecks. [45], [46] |
| **2. "Data-Centric AI" for RDL** | Data-Centric AI | Medium | **Problem**: The "RDB-to-graph" conversion is a critical, high-variance design choice, not just pre-processing. [72], [73] **Opportunity**: Develop a framework for "automated graph schema engineering" that learns the optimal graph representation for a given task. |
| **3. True, End-to-End Multi-Modal RDL** | Architecture | High | **Problem**: Real RDBs are multi-modal (text, images, numbers) [61]. Current SOTA (e.g., Kumo [33]) bolts on an LLM (late-fusion). **Opportunity**: A single, unified model that can natively attend to graph structure, sequential text, and grid-like images. [79] |
| **4. The Neuro-Symbolic Database** | Neuro-Symbolic AI | High / "Holy Grail" | **Problem**: A fork exists between sub-symbolic (Griffin [24]), set-centric (RT [21]), and symbolic (DBFormer [26]) approaches. **Opportunity**: Unify them. A "neural query processor" that can jointly learn to write an optimal SQL query (symbolic) [81] and perform the predictive reasoning (sub-symbolic). |
| **5. True "Zero-Shot" RDB Generalization** | Foundation Models | High | **Problem**: The "Generalization Chasm." [43] How do you pretrain a model to generalize to a completely unseen database schema? **Opportunity**: Develop novel pretraining tasks (beyond Griffin's [24]) that teach a model abstract relational concepts (e.g., "many-to-many") independent of column names. [44] |
| **6. Trust, Explainability, and Robustness** | AI Safety / XAI | Medium | **Problem**: Evergreen challenges. How do you explain a prediction from 5 tables and 10 hops? [82] How do you guarantee privacy? [80] **Opportunity**: Adapt diffusion models for privacy-preserving synthetic RDBs [39]. Develop GNN explainability methods (e.g., GNNExplainer) for RDL's temporal, heterogeneous structures. [82] |

---

## 🤝 Contributing

This is a living document and a community resource. Contributions are welcome! If you see a new paper, model, or resource that should be included, please open an **Issue** or submit a **Pull Request**.

---

## Works Cited

1.  [[1706.03762] Attention Is All You Need][1] - arXiv (The original Transformer paper)
2.  [[1706.02216] Inductive Representation Learning on Large Graphs][2] - arXiv (The original GraphSAGE paper)
3.  [williamleif/GraphSAGE][3] - GitHub (Official GraphSAGE Code)
4.  [Build a GNN-based real-time fraud detection solution...][4] - Amazon AWS Blog
5.  [Build a GNN-based real-time fraud detection solution using Amazon SageMaker...][5] - Amazon AWS Blog (Neptune)
6.  [How AWS uses graph neural networks to meet customer needs][6] - Amazon Science
7.  [Analysing Privacy Policies... case studies of Tinder and Spotify][7] - NIH
8.  [PyTorch Geometric Documentation][8] - Read the Docs
9.  [pyg-team/pytorch_geometric: Graph Neural Network Library for PyTorch][9] - GitHub
10. [Do Transformers Really Perform Badly for Graph Representation?][10] - NeurIPS (The Graphormer paper)
11. [[2312.04615] Relational Deep Learning: Graph Representation Learning on Relational Databases][11] - arXiv
12. [ICML Poster Position: Relational Deep Learning][12] - ICML 2024
13. [Relational Deep Learning: Challenges, Foundations and Next-Generation Architectures][13] - arXiv (2506.16654)
14. [RelBench: Relational Deep Learning Benchmark][14] - Official Website
15. [snap-stanford/relbench: RelBench: Relational Deep Learning Benchmark][15] - GitHub
16. [[2407.20060] RelBench: A Benchmark for Deep Learning on Relational Databases][16] - arXiv
17. [[2003.01332] Heterogeneous Graph Transformer][17] - arXiv (The HGT paper)
18. [acbull/HGT-DGL: Code for "Heterogeneous Graph Transformer" (DGL)][18] - GitHub
19. [acbull/pyHGT: Code for "Heterogeneous Graph Transformer" (PyG)][19] - GitHub
20. [[2505.10960] Relational Graph Transformer][20] - arXiv (The RelGT paper)
21. [[2510.06377] A Relational Transformer for Relational Deep Learning][21] - arXiv (The RT paper)
22. [snap-stanford/relational-transformer: Official code for Relational Transformer][22] - GitHub
23. [[2511.04557] Integrating Temporal and Structural Context in Graph Transformers for RDL][23] - arXiv (The RGP paper)
24. [[2505.05568] Griffin: Towards a Graph-Centric Relational Database Foundation Model][24] - arXiv
25. [yanxwb/Griffin: Official implementation of... “Griffin”][25] - GitHub
26. [[2412.05218] Transformers Meet Relational Databases][26] - arXiv (The DBFormer paper)
27. [Kumo: The Ultimate Guide to Predictive AI on Relational Data][27] - Skywork
28. [KumoRFM: A Foundation Model for In-Context Learning on Relational Data (PDF)][28] - Kumo.ai
29. [Kumo: AI models for relational data][29] - Kumo.ai Homepage
30. [Kumo AI SaaS Security White Paper][30]
31. [Architecture and Security - Kumo Documentation][31]
32. [Graph Neural Networks (GNNs): Introduction and examples][32] - Kumo.ai
33. [Enabling Kumo's predictive AI natively in Snowflake][33] - Kumo.ai
34. [What model architectures does Kumo incorporate...][34] - Kumo Docs
35. [Hybrid Graph Neural Networks][35] - Kumo.ai
36. [Relational Graph Transformers: A New Frontier...][36] - Kumo.ai
37. [Growth and Marketing Examples][37] - Kumo.ai
38. [Improving recommendation systems with LLMs and Graph Transformers][38] - Kumo.ai
39. [Diffusion-Model-Leiden/awesome-diffusion-models-for-tabular-data][39] - GitHub
40. [(PDF) RelDiff: Relational Data Generative Modeling...][40] - ResearchGate
41. [Relational Data Generation with Graph Neural Networks and Latent Diffusion Models (PDF)][41] - OpenReview
42. [[2506.00710] RelDiff: Relational Data Generative Modeling...][42] - arXiv
43. [Graph foundation models for relational data][43] - Google Research Blog
44. [[2305.15321] Towards Foundation Models for Relational Databases [Vision Paper]][44] - arXiv
45. [Open Problems in Graph Neural Networks][45] - Medium
46. [Challenges and Trade-Offs for Graph Neural Networks][46] - Oregon State
47. [On the Bottleneck of Graph Neural Networks and its Practical Implications][47] - OpenReview
48. [How Transformers Work: A Detailed Exploration...][48] - DataCamp
49. [“Attention is all you need” in plain English][49] - Medium
50. [Attention Is All You Need — And All We’ve Lost][50] - Medium
51. [LLM Transformer Model Visually Explained][51] - Polo Club of Data Science
52. [Deep Graph Library (DGL) Homepage][52]
53. [dmlc/dgl: Python package built to ease deep learning on graph...][53] - GitHub
54. [PyTorch Geometric vs Deep Graph Library][54] - Medium
55. [7 Open Source Libraries for Deep Learning Graphs][55] - Exxact Corporation
56. [[D] Are GNNs obsolete because of transformers?][56] - Reddit
57. [Comparative Analysis of GNNs and Transformers for Robust Fake News Detection][57] - MDPI
58. [An Introduction to Graph Transformers][58] - Kumo.ai
59. [Graph Transformer — pytorch_geometric documentation][59]
60. [Graph Transformer (Tutorial)][60] - Medium
61. [The Relational Awakening: Pair-Wise GNNs to Transformer Backbones...][61] - Medium
62. [Self-Service ML with Relational Deep Learning][62] - Medium
63. [RelGNN: Composite Message Passing for Relational Deep Learning][63] - GitHub
64. [Heterogeneous Graph Generation: A Hierarchical Approach...][64] - arXiv
65. [Relative performance drop (%) when position encoding (PE) is removed...][65] - ResearchGate (HGT+PE Analysis)
66. [Runtime Comparison of HGT and HGT+PE baseline...][66] - ResearchGate (HGT+PE Analysis)
67. [lucidrains/perceiver-pytorch: Implementation of Perceiver...][67] - GitHub
68. [krasserm/perceiver-io: A PyTorch implementation of Perceiver...][68] - GitHub
69. [Rishit-dagli/Perceiver: Implementation of Perceiver...][69] - GitHub
70. [Google's Griffin Architecture: Fast Inference...][70] - (Disambiguation: This is the *LLM* Griffin, not the RDL model)
71. [DBFormer: A Dual-Branch Adaptive Remote Sensing Image...][71] - (Disambiguation: This is a *segmentation* model, not the RDL model)
72. [A Data-Centric AI Paradigm for Socio-Industrial and Global Challenges][72] - MDPI
73. [Data-Centric AI vs. Model-Centric AI][73] - Data-Centric AI (MIT)
74. [[2209.15421] TabDDPM: Modelling Tabular Data with Diffusion Models][74] - arXiv
75. [TabDDPM: Modelling Tabular Data with Diffusion Models][75] - PMLR
76. [When to Use GenAI Versus Predictive AI][76] - MIT Sloan
77. [TabDiff: a Mixed-type Diffusion Model for Tabular Data Generation][77] - OpenReview
78. [[2401.16755] Diffusion Model for Relational Inference][78] - arXiv
79. [[2412.14515] Relational Programming with Foundation Models][79] - arXiv
80. [On the Opportunities and Risks of Foundation Models][80] - Stanford CRFM
81. [[2502.12908] Graph Neural Networks for Databases: A Survey][81] - arXiv
82. [Graph Machine Learning Explainability with PyG][82] - Medium

---

[1]: https://arxiv.org/abs/1706.03762
[2]: https://arxiv.org/abs/1706.02216
[3]: https://github.com/williamleif/GraphSAGE
[4]: https://aws.amazon.com/blogs/machine-learning/build-a-gnn-based-real-time-fraud-detection-solution-using-the-deep-graph-library-without-using-external-graph-storage/
[5]: https://aws.amazon.com/blogs/machine-learning/build-a-gnn-based-real-time-fraud-detection-solution-using-amazon-sagemaker-amazon-neptune-and-the-deep-graph-library/
[6]: https://www.amazon.science/blog/how-aws-uses-graph-neural-networks-to-meet-customer-needs
[7]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11459776/
[8]: https://pytorch-geometric.readthedocs.io/
[9]: https://github.com/pyg-team/pytorch_geometric
[10]: https://proceedings.neurips.cc/paper/2021/hash/f1c1592588411002af340cbaedd6fc33-Abstract.html
[11]: https://arxiv.org/abs/2312.04615
[12]: https://icml.cc/virtual/2024/poster/34733
[13]: https://arxiv.org/html/2506.16654v1
[14]: https://relbench.stanford.edu/
[15]: https://github.com/snap-stanford/relbench
[16]: https://arxiv.org/abs/2407.20060
[17]: https://arxiv.org/abs/2003.01332
[18]: https://github.com/acbull/HGT-DGL
[19]: https://github.com/acbull/pyHGT
[20]: https://arxiv.org/abs/2505.10960
[21]: https://arxiv.org/abs/2510.06377
[22]: https://github.com/snap-stanford/relational-transformer
[23]: https://arxiv.org/html/2511.04557v1
[24]: https://arxiv.org/abs/2505.05568
[25]: https://github.com/yanxwb/Griffin
[26]: https://arxiv.org/pdf/2412.05218
[27]: https://skywork.ai/skypage/en/Kumo-The-Ultimate-Guide-to-Predictive-AI-on-Relational-Data/1976549405707268096
[28]: https://kumo.ai/research/kumo_relational_foundation_model.pdf
[29]: https://kumo.ai/
[30]: https://kumo.ai/docs/security-and-governance/
[31]: https://docs.kumo.ai/docs/architecture-and-security
[32]: https://kumo.ai/research/graph-neural-networks-gnn/
[33]: https://kumo.ai/company/news/enabling-kumos-predictive-ai-natively-in-snowflake/
[34]: https://kumo.ai/docs/troubleshooting/what-model-architectures-does-kumo-incorporate-into-its-gnn-design-space/
[35]: https://kumo.ai/research/hybrid-graph-neural-networks/
[36]: https://kumo.ai/research/relational-graph-transformers/
[37]: https://kumo.ai/docs/examples/growthmarketing/
[38]: https://kumo.ai/research/recommendation-systems-llms-graph-transformers/
[39]: https://github.com/Diffusion-Model-Leiden/awesome-diffusion-models-for-tabular-data
[40]: https://www.researchgate.net/publication/392334481_RelDiff_Relational_Data_Generative_Modeling_with_Graph-Based_Diffusion_Models
[41]: https://openreview.net/pdf?id=MNLR2NYN2Z
[42]: https://arxiv.org/html/2506.00710v1
[43]: https://research.google/blog/graph-foundation-models-for-relational-data/
[44]: https://arxiv.org/abs/2305.15321
[45]: https://medium.com/@nikitaparate9/open-problems-in-graph-neural-networks-58e65bf044c
[46]: https://engineering.oregonstate.edu/events/challenges-and-trade-offs-graph-neural-networks
[47]: https://openreview.net/forum?id=i80OPhOCVH2
[48]: https://www.datacamp.com/tutorial/how-transformers-work
[49]: https://medium.com/@ujwaltickoo/attention-is-all-you-need-in-plain-english-b176d1ad4ada
[50]: https://medium.com/@amjad.shaikh/attention-is-all-you-need-and-all-weve-lost-bf0d38237628
[51]: https://poloclub.github.io/transformer-explainer/
[52]: https://www.dgl.ai/
[53]: https://github.com/dmlc/dgl
[54]: https://medium.com/@khang.pham.exxact/pytorch-geometric-vs-deep-graph-library-626ff1e802
[55]: https://www.exxactcorp.com/blog/Deep-Learning/open-source-libraries-for-deep-learning-graphs
[56]: https://www.reddit.com/r/MachineLearning/comments/1jgwjjk/d_are-gnns-obsolete-because-of-transformers/
[57]: https://www.mdpi.com/2079-9292/13/23/4784
[58]: https://kumo.ai/research/introduction-to-graph-transformers/
[59]: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/graph_transformer.html
[60]: https://medium.com/@reutdayan1/graph-transformer-2ede65db4658
[61]: https://medium.com/dataai/the-relational-awakening-pair-wise-gnns-to-transformer-backbones-contextgnn-relgt-2403ecaaeace
[62]: https://medium.com/data-science/self-service-ml-with-relational-deep-learning-beb693a21d5b
[63]: https://github.com/snap-stanford/RelGNN
[64]: https://arxiv.org/html/2410.11972
[65]: https://www.researchgate.net/figure/Relative-performance-drop-when-position-encoding-PE-is-removed-from-HGT-PE-models_tbl4_391856673
[66]: https://www.researchgate.net/figure/Runtime-Comparison-of-HGT-and-HGT-PE-baseline-Adding-the-Laplacian-Positional-Encoding_fig1_391856673
[67]: https://github.com/lucidrains/perceiver-pytorch
[68]: https://github.com/krasserm/perceiver-io
[69]: https://github.com/Rishit-dagli/Perceiver
[70]: https://www.youtube.com/watch?v=Zkfqr0AmGGA
[71]: https://www.researchgate.net/publication/393080176_DBFormer_A_Dual-Branch_Adaptive_Remote_Sensing_Image_Resolution_Fine-Grained_Weed_Segmentation_Network
[72]: https://www.mdpi.com/2079-9292/13/11/2156
[73]: https://dcai.csail.mit.edu/2024/data-centric-model-centric/
[74]: https://arxiv.org/html/2209.15421v2
[75]: https://proceedings.mlr.press/v202/kotelnikov23a/kotelnikov23a.pdf
[76]: https://sloanreview.mit.edu/article/when-to-use-genai-versus-predictive-ai/
[77]: https://openreview.net/forum?id=swvURjrt8z
[78]: https://arxiv.org/html/2401.16755v2
[79]: https://arxiv.org/html/2412.14515v1
[80]: https://crfm.stanford.edu/report.html
[81]: https://arxiv.org/html/2502.12908v1
[82]: https://medium.com/@pytorch_geometric/graph-machine-learning-explainability-with-pyg-ff13cffc23c2
