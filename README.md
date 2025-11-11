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

## Works Cited

## Works Cited

[1]: https://arxiv.org/abs/1706.03762
[2]: https://papers.nips.cc/paper/7181-attention-is-all-you-need
[3]: https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf
[4]: https://snap.stanford.edu/graphsage/
[5]: https://arxiv.org/abs/1706.02216
[6]: https://github.com/williamleif/GraphSAGE
[7]: https://aws.amazon.com/blogs/machine-learning/build-a-gnn-based-real-time-fraud-detection-solution-using-the-deep-graph-library-without-using-external-graph-storage/
[8]: https://www.amazon.science/blog/how-aws-uses-graph-neural-networks-to-meet-customer-needs
[9]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11459776/
[10]: https://pytorch-geometric.readthedocs.io/
[11]: https://github.com/pyg-team/pytorch_geometric
[12]: https://proceedings.neurips.cc/paper/2021/hash/f1c1592588411002af340cbaedd6fc33-Abstract.html
[13]: https://arxiv.org/abs/2312.04615
[14]: https://icml.cc/virtual/2024/poster/34733
[15]: https://relbench.stanford.edu/
[16]: https://github.com/snap-stanford/relbench
[17]: https://arxiv.org/abs/2003.01332
[18]: https://arxiv.org/abs/2505.10960
[19]: https://arxiv.org/html/2511.04557v1
[20]: https://arxiv.org/abs/2505.05568
[21]: https://arxiv.org/pdf/2412.05218
[22]: https://skywork.ai/skypage/en/Kumo-The-Ultimate-Guide-to-Predictive-AI-on-Relational-Data/1976549405707268096
[23]: https://kumo.ai/research/kumo_relational_foundation_model.pdf
[24]: https://github.com/Diffusion-Model-Leiden/awesome-diffusion-models-for-tabular-data
[25]: https://www.researchgate.net/publication/392334481_RelDiff_Relational_Data_Generative_Modeling_with_Graph-Based_Diffusion_Models
[26]: https://research.google/blog/graph-foundation-models-for-relational-data/
[27]: https://medium.com/@nikitaparate9/open-problems-in-graph-neural-networks-58e65bf044c
[28]: https://openreview.net/forum?id=i80OPhOCVH2
[29]: https://www.datacamp.com/tutorial/how-transformers-work
[30]: https://medium.com/@ujwaltickoo/attention-is-all-you-need-in-plain-english-b176d1ad4ada
[31]: https://medium.com/@amjad.shaikh/attention-is-all-you-need-and-all-weve-lost-bf0d38237628
[32]: https://poloclub.github.io/transformer-explainer/
[33]: https://www.dgl.ai/
[34]: https://github.com/dmlc/dgl
[35]: https://pytorch-geometric.readthedocs.io/en/2.6.0/modules/datasets.html
[36]: https://medium.com/@khang.pham.exxact/pytorch-geometric-vs-deep-graph-library-626ff1e802
[37]: https://www.exxactcorp.com/blog/Deep-Learning/open-source-libraries-for-deep-learning-graphs
[38]: https://aws.amazon.com/blogs/machine-learning/build-a-gnn-based-real-time-fraud-detection-solution-using-amazon-sagemaker-amazon-neptune-and-the-deep-graph-library/
[39]: https://www.reddit.com/r/MachineLearning/comments/1jgwjjk/d_are_gnns_obsolete_because_of_transformers/
[40]: https://www.mdpi.com/2079-9292/13/23/4784
[41]: https://kumo.ai/research/introduction-to-graph-transformers/
[42]: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/graph_transformer.html
[43]: https://medium.com/@reutdayan1/graph-transformer-2ede65db4658
[44]: https://medium.com/dataai/the-relational-awakening-pair-wise-gnns-to-transformer-backbones-contextgnn-relgt-2403ecaaeace
[45]: https://medium.com/data-science/self-service-ml-with-relational-deep-learning-beb693a21d5b
[46]: https://arxiv.org/html/2506.16654v1
[47]: https://arxiv.org/abs/2407.20060
[48]: https://github.com/snap-stanford/RelGNN
[49]: https://kumo.ai/research/relbench/
[50]: https://arxiv.org/html/2410.11972
[51]: https://arxiv.org/html/2407.13158v1
[52]: https://arxiv.org/html/2505.10960v1
[53]: https://www.researchgate.net/figure/Relative-performance-drop-when-position-encoding-PE-is-removed-from-HGT-PE-models_tbl4_391856673
[54]: https://www.researchgate.net/figure/Runtime-Comparison-of-HGT-and-HGT-PE-baseline-Adding-the-Laplacian-Positional-Encoding_fig1_391856673
[55]: https://github.com/acbull/HGT-DGL
[56]: https://github.com/acbull/pyHGT
[57]: https://arxiv.org/abs/2511.04557
[58]: https://openreview.net/pdf?id=fcVIJ2WSIX
[59]: https://openreview.net/pdf/f3cde5eb6202e1a6d509de5b87ca7fd1e074790e.pdf
[60]: https://www.themoonlight.io/de/review/integrating-temporal-and-structural-context-in-graph-transformers-for-relational-deep-learning
[61]: https://www.themoonlight.io/review/integrating-temporal-and-structural-context-in-graph-transformers-for-relational-deep-learning
[62]: https://www.researchgate.net/publication/397365799_Integrating_Temporal_and_Structural_Context_in_Graph_Transformers_for_Relational_Deep_Learning
[63]: https://github.com/Rishit-dagli/Perceiver
[64]: https://github.com/krasserm/perceiver-io
[65]: https://github.com/lucidrains/perceiver-pytorch
[66]: https://www.youtube.com/watch?v=Zkfqr0AmGGA
[67]: https://github.com/yanxwb/Griffin
[68]: https://arxiv.org/html/2505.05568v2
[69]: https://assets.amazon.science/5f/e5/27c801c445cfa9d8d3a167f47861/griffin-towards-a-graph-centric-relational-database-foundation-model.pdf
[70]: https://www.researchgate.net/publication/391657707_Griffin_Towards_a_Graph-Centric_Relational_Database_Foundation_Model
[71]: https://github.com/yanxwb/Griffin
[72]: https://www.researchgate.net/publication/393080176_DBFormer_A_Dual-Branch_Adaptive_Remote_Sensing_Image_Resolution_Fine-Grained_Weed_Segmentation_Network
[73]: https://www.semanticscholar.org/paper/DBFormer%3A-A-Dual-Branch-Adaptive-Remote-Sensing-She-Tang/d3bda59cb5a40e1b19ee1572ceabdb689f779d2e
[74]: https://www.mdpi.com/2072-4292/17/13/2203
[75]: https://arxiv.org/html/2412.05218v1
[76]: https://chatpaper.com/paper/88358
[77]: https://www.mdpi.com/2079-9292/13/11/2156
[78]: https://kumo.ai/
[79]: https://kumo.ai/docs/security-and-governance/
[80]: https://docs.kumo.ai/docs/architecture-and-security
[81]: https://kumo.ai/research/graph-neural-networks-gnn/
[82]: https://kumo.ai/company/news/enabling-kumos-predictive-ai-natively-in-snowflake/
[83]: https://kumo.ai/docs/troubleshooting/what-model-architectures-does-kumo-incorporate-into-its-gnn-design-search-space/
[84]: https://kumo.ai/research/hybrid-graph-neural-networks/
[85]: https://kumo.ai/research/relational-graph-transformers/
[86]: https://kumo.ai/docs/examples/growthmarketing/
[87]: https://kumo.ai/research/recommendation-systems-llms-graph-transformers/
[88]: https://arxiv.org/html/2209.15421v2
[89]: https://proceedings.mlr.press/v202/kotelnikov23a/kotelnikov23a.pdf
[90]: https://sloanreview.mit.edu/article/when-to-use-genai-versus-predictive-ai/
[91]: https://openreview.net/forum?id=swvURjrt8z
[92]: https://openreview.net/pdf?id=MNLR2NYN2Z
[93]: https://arxiv.org/html/2506.00710v1
[94]: https://arxiv.org/html/2401.16755v2
[95]: https://arxiv.org/abs/2305.15321
[96]: https://engineering.oregonstate.edu/events/challenges-and-trade-offs-graph-neural-networks
[97]: https://arxiv.org/html/2412.14515v1
[98]: https://crfm.stanford.edu/report.html
[99]: https://www.reddit.com/r/machinelearningnews/comments/1egf3z2/researchers_at_stanford_present_relbench_an_open/
[100]: https://dcai.csail.mit.edu/2024/data-centric-model-centric/
[101]: https://arxiv.org/html/2507.16991v1
[102]: https://arxiv.org/html/2502.12908v1
[103]: https://medium.com/@pytorch_geometric/graph-machine-learning-explainability-with-pyg-ff13cffc23c2

