# The Relational Awakening: Foundational Models for Structured Data

This repository is a one-stop shop and community-maintained resource for the emerging field of **Relational Deep Learning (RDL)**. It is based on the report, "The Relational Awakening: From Parallel Paths to Foundational Models for Structured Data."

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
| **Unstructured Data (Sequences)** | **"Attention is All You Need"** [1] | Introduced the **Transformer**, which uses a self-attention mechanism. It's highly parallelizable and captures long-range dependencies [46], igniting the LLM boom [47], [49]. |
| **Structured Data (Graphs)** | **"Inductive Representation Learning on Large Graphs"** [2] | Introduced **GraphSAGE**, an *inductive* framework that learns functions to aggregate neighborhood features. This allowed GNNs to generalize to unseen nodes and scale to massive industrial graphs. [Code: [3]] |

This split created two parallel research tracks: one for the unstructured public internet (LLMs) and one for structured, proprietary business data (GNNs).

## II. The GNN "Underground": Industrial Adoption

While LLMs captured public attention, GNNs were quietly operationalized for high-value enterprise tasks. This was driven by a new generation of open-source libraries.

### Key Enablers (Open-Source Libraries)

* **PyTorch Geometric (PyG)** [8], [9]: A library for deep learning on graphs, providing scalable mini-batch loaders, multi-GPU support, and first-class support for heterogeneous graphs.
* **Deep Graph Library (DGL)** [50], [51]: A framework-agnostic library (supporting PyTorch, TensorFlow, etc.) for easing deep learning on graphs.

### Industrial Applications

Armed with these tools, companies like **Amazon** [4], [6] and **Spotify** [7] deployed GNNs at scale for:
* **Fraud Detection**: Modeling financial transactions as heterogeneous graphs (e.g., nodes for transactions, devices, IPs) to perform node classification (fraudulent or not). [4], [5]
* **Recommendation Engines**: Treating users and items as a bipartite graph to learn rich embeddings for a user based on the items they've interacted with (and vice-versa).

This industrial use of heterogeneous GNNs was an early, ad-hoc form of Relational Deep Learning.

## III. The Bridge: Fusing Attention and Structure

Traditional message-passing GNNs hit a "glass ceiling" due to several key limitations:
* **Over-smoothing**: As layers get deeper, node embeddings converge to a single, indistinct value. [56]
* **Over-squashing**: Information from exponentially large neighborhoods is "squashed" into a fixed-size vector, creating a bottleneck. [45]
* **Long-Range Dependencies**: Capturing information between distant nodes requires many layers, which triggers the problems above. [56]

The **Graph Transformer** emerged as the solution, a hybrid architecture that fuses the global reasoning of Transformers with the structural awareness of GNNs. Models like **Graphormer** [10] and **GraphGPS** [57] proved that a "stacking" of GNN layers (for local bias) and Transformer layers (for global reasoning) is the state-of-the-art.

### GNN vs. Graph Transformer: A Comparative Framework

| Aspect | Graph Neural Networks (GNNs) | Graph Transformers (GTs) |
| :--- | :--- | :--- |
| **Information Flow** | Sequential message passing between connected nodes [56] | Self-attention; direct aggregation from all nodes [56] |
| **Long-Range Dependencies** | Struggle; require many layers (hops) [56] | Captured effectively in a single layer [56] |
| **Over-Smoothing** | Prone with deep layers; embeddings become too similar [56] | Less prone; attention allows flexible information mixing [56] |
| **Over-Squashing** | Prone; information compressed when aggregating many neighbors [45] | Mitigated by attention, distributing information effectively [56] |
| **Scalability** | Efficient; typically linear in number of nodes + edges | Poor; standard attention is $O(N^2)$ in number of nodes [58] |
| **Inductive Bias** | Strong; hard-coded for locality and graph topology | Weak; assumes a fully connected graph by default |

## IV. The Relational Awakening: Formalizing RDL

The ad-hoc industrial work and new hybrid architectures coalesced into a formal field: **Relational Deep Learning (RDL)** [11], [12], [13].

The RDL blueprint is a simple, powerful idea:
1.  **Nodes**: Create a node for each **row** in each table.
2.  **Edges**: Create edges based on **primary-foreign key links**.
3.  **Graph Type**: The result is inherently a **temporal, heterogeneous graph** [13], [59], [60].

This formalization highlighted a "generalization chasm": a GNN trained on one database schema couldn't be used on another. [41] This created the need for a standardized benchmark to build and test generalist models.

### The "ImageNet" for RDL: RelBench

The **Relational Deep Learning Benchmark (RelBench)** [14] was developed by Stanford to fill this need. It is the foundational infrastructure for RDL research, providing:
* **Realistic Datasets**: Large-scale, diverse datasets from e-commerce, social networks, etc.
* **Standardized Tasks**: 30 challenging, domain-specific predictive tasks.
* **Unified Pipelines**: Standard data splits, evaluation pipelines, and baselines [15], [16].

**Links:**
* **Website:** [relbench.stanford.edu][14]
* **GitHub:** [github.com/snap-stanford/relbench][15]

RDL models trained on RelBench were shown to **reduce model development time by over 95%** by automating the time-consuming process of manual feature engineering. [13], [16]

## V. Architectures for a Relational Foundation

The creation of RelBench spurred a new wave of SOTA architectures. Here is a comparative analysis of the key models.

### Comparative Analysis of Modern Relational Models

| Model | Core Architecture | Key Innovation | Handling of RDBs | Open Source |
| :--- | :--- | :--- | :--- | :--- |
| **HGT+PE** [17], [20] | Heterogeneous Transformer + Laplacian PE | Baseline GT for heterogeneous graphs. | Converts RDB to graph; uses expensive, general PE for structure. [63], [64] | **Yes** [18], [19] |
| **RelGT** [20] | Hybrid (Local+Global) Graph Transformer | **Multi-element tokenization** (type, hop, time, structure) instead of one expensive PE. | Converts RDB to graph; time is a feature. | **Yes** [20] |
| **RGP** [21] | Perceiver-based Graph Transformer | **Active temporal sampler** + struct/temporal fusion. Treats time as an active sampling dimension. | Converts RDB to graph; time is an active sampling dimension. | Not Specified |
| **Griffin** [22] | MPNN (GNN) + Cross-Attention | **Foundation Model Pretraining** (masked value completion) for transfer learning. | Converts RDB to graph; pretrains for transferability. | **Yes** [23] |
| **DBFormer** [24] | Relational-native Transformer | **"Graph-less" approach**. Message passing based on the "formal relational model." | **Does not convert to graph**; models relations directly. | **Yes** [24] |

This analysis reveals a fundamental fork in the road:
* **Graph-Centric Path (Griffin, RelGT)**: Convert the database *to* a graph, then apply a Graph Transformer.
* **Relational-Centric Path (DBFormer)**: Build a new Transformer that *natively* understands relational algebra.

## VI. The Industrialization of RDL: Kumo.ai

The **Kumo.ai** platform [25], [27] is a case study in how RDL concepts are being productized. It provides a SaaS "easy button" that connects to a data warehouse (e.g., Snowflake) and automates the entire RDL pipeline. [28], [29]

Kumo's core technology, the KumoRFM [26], is not a single model but a hybrid **"mixture-of-experts"** platform. It maintains a toolbox of SOTA architectures and selects the right one for the task:
* **Classic GNNs**: Uses **GraphSAGE** for cold-start problems and **GIN** for frequency signals. [30], [32]
* **Hybrid GNNs**: Deploys custom hybrids for tasks like recommendations. [33]
* **Relational Graph Transformers (RGTs)**: Uses RGTs with time embeddings to handle complex, temporal data. [34]
* **LLM Integration**: Integrates GNNs with LLMs (like Snowflake Cortex) to incorporate unstructured data (e.g., product descriptions). [31], [36]

Kumo.ai demonstrates that the industrial "foundation model" is a practical, hybrid AutoML platform that aggregates the entire RDL research field.

## VII. A Generative Detour: Diffusion Models

Can diffusion models be used for predictions on relational data?
**The answer is primarily *indirectly***. Diffusion models are *generative* [74] and are used to support downstream *predictive* models (like GNNs or XGBoost).

Their two main uses are: [37]
1.  **Data Augmentation**: Models like **TabDDPM** [72], [73] generate new, synthetic tabular data. This is most often used to **address class imbalance** (e.g., generating more "fraud" samples for a classifier). [38]
2.  **Data Imputation**: Models like **CSDI** fill in missing or incomplete entries [37], allowing predictive models to be trained on cleaner data.

This is now being extended to multi-table data with models like **RelDiff** [38], [39], [40], which generates entire synthetic relational databases.

### Analysis of Diffusion Models for Tabular/Relational Data

| Model | Primary Task | Use Case | Open Source |
| :--- | :--- | :--- | :--- |
| **TabDDPM** [72] | Generative | Data Augmentation (for prediction) | **Yes** |
| **TabDiff** [75] | Generative | Data Augmentation, Imputation | **Yes** |
| **RelDiff** [40] | Generative | Relational Data Synthesis (for prediction) | **Yes** |
| **DiffRI** [76] | Predictive | Relational Inference (Time Series) | Not Specified |

## VIII. The New Frontier: Open Problems

The convergence of structured data, GNNs, and Transformers has opened a new frontier. The field is moving from task-specific models to general-purpose foundation models. The following roadmap outlines the key open challenges and opportunities.

### Future Opportunities and Open Problems in Relational AI

| Opportunity / Open Problem | Domain | Difficulty | Description & Rationale |
| :--- | :--- | :--- | :--- |
| **1. GNN Architectural Limits** | Architecture | Medium | **Problem**: Over-smoothing [56] and over-squashing [45] remain fundamental bottlenecks in the GNN components of hybrid models. **Opportunity**: Develop more expressive GNN aggregators that can capture long-range info without these bottlenecks. [43], [44] |
| **2. "Data-Centric AI" for RDL** | Data-Centric AI | Medium | **Problem**: The "RDB-to-graph" conversion is a critical, high-variance design choice, not just pre-processing. [70], [71] **Opportunity**: Develop a framework for "automated graph schema engineering" that learns the optimal graph representation for a given task. |
| **3. True, End-to-End Multi-Modal RDL** | Architecture | High | **Problem**: Real RDBs are multi-modal (text, images, numbers) [59]. Current SOTA (e.g., Kumo [31]) bolts on an LLM (late-fusion). **Opportunity**: A single, unified model that can natively attend to graph structure, sequential text, and grid-like images. [77] |
| **4. The Neuro-Symbolic Database** | Neuro-Symbolic AI | High / "Holy Grail" | **Problem**: A fork exists between sub-symbolic (Griffin [22]) and symbolic (DBFormer [24]) approaches. **Opportunity**: Unify them. A "neural query processor" that can jointly learn to write an optimal SQL query (symbolic) [79] and perform the predictive reasoning (sub-symbolic). |
| **5. True "Zero-Shot" RDB Generalization** | Foundation Models | High | **Problem**: The "Generalization Chasm." [41] How do you pretrain a model to generalize to a completely unseen database schema? **Opportunity**: Develop novel pretraining tasks (beyond Griffin's [22]) that teach a model abstract relational concepts (e.g., "many-to-many") independent of column names. [42] |
| **6. Trust, Explainability, and Robustness** | AI Safety / XAI | Medium | **Problem**: Evergreen challenges. How do you explain a prediction from 5 tables and 10 hops? [80] How do you guarantee privacy? [78] **Opportunity**: Adapt diffusion models for privacy-preserving synthetic RDBs [37]. Develop GNN explainability methods (e.g., GNNExplainer) for RDL's temporal, heterogeneous structures. [80] |

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
21. [[2511.04557] Integrating Temporal and Structural Context in Graph Transformers for RDL][21] - arXiv (The RGP paper)
22. [[2505.05568] Griffin: Towards a Graph-Centric Relational Database Foundation Model][22] - arXiv
23. [yanxwb/Griffin: Official implementation of... “Griffin”][23] - GitHub
24. [[2412.05218] Transformers Meet Relational Databases][24] - arXiv (The DBFormer paper)
25. [Kumo: The Ultimate Guide to Predictive AI on Relational Data][25] - Skywork
26. [KumoRFM: A Foundation Model for In-Context Learning on Relational Data (PDF)][26] - Kumo.ai
27. [Kumo: AI models for relational data][27] - Kumo.ai Homepage
28. [Kumo AI SaaS Security White Paper][28]
29. [Architecture and Security - Kumo Documentation][29]
30. [Graph Neural Networks (GNNs): Introduction and examples][30] - Kumo.ai
31. [Enabling Kumo's predictive AI natively in Snowflake][31] - Kumo.ai
32. [What model architectures does Kumo incorporate...][32] - Kumo Docs
33. [Hybrid Graph Neural Networks][33] - Kumo.ai
34. [Relational Graph Transformers: A New Frontier...][34] - Kumo.ai
35. [Growth and Marketing Examples][35] - Kumo.ai
36. [Improving recommendation systems with LLMs and Graph Transformers][36] - Kumo.ai
37. [Diffusion-Model-Leiden/awesome-diffusion-models-for-tabular-data][37] - GitHub
38. [(PDF) RelDiff: Relational Data Generative Modeling...][38] - ResearchGate
39. [Relational Data Generation with Graph Neural Networks and Latent Diffusion Models (PDF)][39] - OpenReview
40. [[2506.00710] RelDiff: Relational Data Generative Modeling...][40] - arXiv
41. [Graph foundation models for relational data][41] - Google Research Blog
42. [[2305.15321] Towards Foundation Models for Relational Databases [Vision Paper]][42] - arXiv
43. [Open Problems in Graph Neural Networks][43] - Medium
44. [Challenges and Trade-Offs for Graph Neural Networks][44] - Oregon State
45. [On the Bottleneck of Graph Neural Networks and its Practical Implications][45] - OpenReview
46. [How Transformers Work: A Detailed Exploration...][46] - DataCamp
47. [“Attention is all you need” in plain English][47] - Medium
48. [Attention Is All You Need — And All We’ve Lost][48] - Medium
49. [LLM Transformer Model Visually Explained][49] - Polo Club of Data Science
50. [Deep Graph Library (DGL) Homepage][50]
51. [dmlc/dgl: Python package built to ease deep learning on graph...][51] - GitHub
52. [PyTorch Geometric vs Deep Graph Library][52] - Medium
53. [7 Open Source Libraries for Deep Learning Graphs][53] - Exxact Corporation
54. [[D] Are GNNs obsolete because of transformers?][54] - Reddit
55. [Comparative Analysis of GNNs and Transformers for Robust Fake News Detection][55] - MDPI
56. [An Introduction to Graph Transformers][56] - Kumo.ai
57. [Graph Transformer — pytorch_geometric documentation][57]
58. [Graph Transformer (Tutorial)][58] - Medium
59. [The Relational Awakening: Pair-Wise GNNs to Transformer Backbones...][59] - Medium
60. [Self-Service ML with Relational Deep Learning][60] - Medium
61. [RelGNN: Composite Message Passing for Relational Deep Learning][61] - GitHub
62. [Heterogeneous Graph Generation: A Hierarchical Approach...][62] - arXiv
63. [Relative performance drop (%) when position encoding (PE) is removed...][63] - ResearchGate (HGT+PE Analysis)
64. [Runtime Comparison of HGT and HGT+PE baseline...][64] - ResearchGate (HGT+PE Analysis)
65. [lucidrains/perceiver-pytorch: Implementation of Perceiver...][65] - GitHub
66. [krasserm/perceiver-io: A PyTorch implementation of Perceiver...][66] - GitHub
67. [Rishit-dagli/Perceiver: Implementation of Perceiver...][67] - GitHub
68. [Google's Griffin Architecture: Fast Inference...][68] - YouTube (Note: This is the *LLM* Griffin, not the RDL model)
69. [DBFormer: A Dual-Branch Adaptive Remote Sensing Image...][69] - ResearchGate (Note: This is a *segmentation* model, not the RDL model)
70. [A Data-Centric AI Paradigm for Socio-Industrial and Global Challenges][70] - MDPI
71. [Data-Centric AI vs. Model-Centric AI][71] - Data-Centric AI (MIT)
72. [[2209.15421] TabDDPM: Modelling Tabular Data with Diffusion Models][72] - arXiv
73. [TabDDPM: Modelling Tabular Data with Diffusion Models][73] - PMLR
74. [When to Use GenAI Versus Predictive AI][74] - MIT Sloan
75. [TabDiff: a Mixed-type Diffusion Model for Tabular Data Generation][75] - OpenReview
76. [[2401.16755] Diffusion Model for Relational Inference][76] - arXiv
77. [[2412.14515] Relational Programming with Foundation Models][77] - arXiv
78. [On the Opportunities and Risks of Foundation Models][78] - Stanford CRFM
79. [[2502.12908] Graph Neural Networks for Databases: A Survey][79] - arXiv
80. [Graph Machine Learning Explainability with PyG][80] - Medium

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
[21]: https://arxiv.org/html/2511.04557v1
[22]: https://arxiv.org/abs/2505.05568
[23]: https://github.com/yanxwb/Griffin
[24]: https://arxiv.org/pdf/2412.05218
[25]: https://skywork.ai/skypage/en/Kumo-The-Ultimate-Guide-to-Predictive-AI-on-Relational-Data/1976549405707268096
[26]: https://kumo.ai/research/kumo_relational_foundation_model.pdf
[27]: https://kumo.ai/
[28]: https://kumo.ai/docs/security-and-governance/
[29]: https://docs.kumo.ai/docs/architecture-and-security
[30]: https://kumo.ai/research/graph-neural-networks-gnn/
[31]: https://kumo.ai/company/news/enabling-kumos-predictive-ai-natively-in-snowflake/
[32]: https://kumo.ai/docs/troubleshooting/what-model-architectures-does-kumo-incorporate-into-its-gnn-design-space/
[33]: https://kumo.ai/research/hybrid-graph-neural-networks/
[34]: https://kumo.ai/research/relational-graph-transformers/
[35]: https://kumo.ai/docs/examples/growthmarketing/
[36]: https://kumo.ai/research/recommendation-systems-llms-graph-transformers/
[37]: https://github.com/Diffusion-Model-Leiden/awesome-diffusion-models-for-tabular-data
[38]: https://www.researchgate.net/publication/392334481_RelDiff_Relational_Data_Generative_Modeling_with_Graph-Based_Diffusion_Models
[39]: https://openreview.net/pdf?id=MNLR2NYN2Z
[40]: https://arxiv.org/html/2506.00710v1
[41]: https://research.google/blog/graph-foundation-models-for-relational-data/
[42]: https://arxiv.org/abs/2305.15321
[43]: https://medium.com/@nikitaparate9/open-problems-in-graph-neural-networks-58e65bf044c
[44]: https://engineering.oregonstate.edu/events/challenges-and-trade-offs-graph-neural-networks
[45]: https://openreview.net/forum?id=i80OPhOCVH2
[46]: https://www.datacamp.com/tutorial/how-transformers-work
[47]: https://medium.com/@ujwaltickoo/attention-is-all-you-need-in-plain-english-b176d1ad4ada
[48]: https://medium.com/@amjad.shaikh/attention-is-all-you-need-and-all-weve-lost-bf0d38237628
[49]: https://poloclub.github.io/transformer-explainer/
[50]: https://www.dgl.ai/
[51]: https://github.com/dmlc/dgl
[52]: https://medium.com/@khang.pham.exxact/pytorch-geometric-vs-deep-graph-library-626ff1e802
[53]: https://www.exxactcorp.com/blog/Deep-Learning/open-source-libraries-for-deep-learning-graphs
[54]: https://www.reddit.com/r/MachineLearning/comments/1jgwjjk/d_are_gnns_obsolete_because_of_transformers/
[55]: https://www.mdpi.com/2079-9292/13/23/4784
[56]: https://kumo.ai/research/introduction-to-graph-transformers/
[57]: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/graph_transformer.html
[58]: https://medium.com/@reutdayan1/graph-transformer-2ede65db4658
[59]: https://medium.com/dataai/the-relational-awakening-pair-wise-gnns-to-transformer-backbones-contextgnn-relgt-2403ecaaeace
[60]: https://medium.com/data-science/self-service-ml-with-relational-deep-learning-beb693a21d5b
[61]: https://github.com/snap-stanford/RelGNN
[62]: https://arxiv.org/html/2410.11972
[63]: https://www.researchgate.net/figure/Relative-performance-drop-when-position-encoding-PE-is-removed-from-HGT-PE-models_tbl4_391856673
[64]: https://www.researchgate.net/figure/Runtime-Comparison-of-HGT-and-HGT-PE-baseline-Adding-the-Laplacian-Positional-Encoding_fig1_391856673
[65]: https://github.com/lucidrains/perceiver-pytorch
[66]: https://github.com/krasserm/perceiver-io
[67]: https://github.com/Rishit-dagli/Perceiver
[68]: https://www.youtube.com/watch?v=Zkfqr0AmGGA
[69]: https://www.researchgate.net/publication/393080176_DBFormer_A_Dual-Branch_Adaptive_Remote_Sensing_Image_Resolution_Fine-Grained_Weed_Segmentation_Network
[70]: https://www.mdpi.com/2079-9292/13/11/2156
[71]: https://dcai.csail.mit.edu/2024/data-centric-model-centric/
[72]: https://arxiv.org/html/2209.15421v2
[73]: https://proceedings.mlr.press/v202/kotelnikov23a/kotelnikov23a.pdf
[74]: https://sloanreview.mit.edu/article/when-to-use-genai-versus-predictive-ai/
[75]: https://openreview.net/forum?id=swvURjrt8z
[76]: https://arxiv.org/html/2401.16755v2
[77]: https://arxiv.org/html/2412.14515v1
[78]: https://crfm.stanford.edu/report.html
[79]: https://arxiv.org/html/2502.12908v1
[80]: https://medium.com/@pytorch_geometric/graph-machine-learning-explainability-with-pyg-ff13cffc23c2
