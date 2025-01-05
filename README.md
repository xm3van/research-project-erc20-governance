# Research Repository: Governance Centralization Using Bipartite Complex Systems Token Network Projections

**Abstract:** Blockchain-based systems, and the platforms built upon them, are frequently governed through tokens granting their holders voting rights over core protocol functions and funds. Centralisation in token-based voting systems, as common in Decentralised Finance (DeFi) protocols, is typically analysed by examining token holdings' distribution across addresses. This paper expands this perspective by exploring shared token holdings of addresses across multiple DeFi protocols. We construct a Statistically Validated Network (SVN) based on shared governance token holding among addresses. Using the links within the SVN, we identify influential addresses that shape these connections and conduct a post-hoc analysis to examine their characteristics and behaviour. Our findings reveal persistent influential links over time, predominantly involving addresses associated with institutional investors who maintain significant token supplies across the sampled protocols. These links often show disproportionate influence in a single token constituting the link. Furthermore, token holdings tend to shift in response to market cycles.

> Citation:  To be added.


**Repository Structure:**

```
project_root/
│
├── data/                   # Directory for raw and processed data files
│   ├── bash_command.txt    # Bash command use for extraction of raw data
│   └── data.zip            # zip file containing raw & processed data files
│
├── src/                    # Source code directory
│   │
│   ├── preprocessing/      # Module for data loading and preprocessing
│   │   ├── data_preprocessor.py # Functions for cleaning, filtering, and preparing data
│   │   └── data_preprocessor_utils.py # Helper functiosn
│   │
│   ├── token_projection/   # Module for generating and analyzing token projection networks
│   │   ├── token_projection_analysis.py # Functions to generate token projection networks
│   │   └── network_analysis.py  # Functions to analyze the generated networks
│   │
│   ├── wallet_projection/  # Module for generating and analyzing wallet projection networks
│   │   └── wallet_network_analysis.py  # Functions to analyze the generated wallet networks
│   │
│   ├── analysis/           # Module for detailed analysis (e.g., clique analysis)
│   │   ├── clique_analysis.py      # Contains the CliqueAnalysis class and related functions
│   │   └── link_analysis.py      # Contains the CliqueAnalysis class and related functions
│   │
│   ├── visualisations/     # Module for creating visualizations of analysis output
│   │
│   └── utilities/          # Common utilities used across the project
│
├── notebooks/              # Jupyter notebooks for experimentation and demonstration
│
└─── outputs/                # directory for research outputs
```


> ***Note**: The research does not cover the analysis of cliques - we encourage the courious to investigate the respective notebooks*