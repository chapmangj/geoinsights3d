---
title: 'GeoInsights 3D: A Python tool for rapid, interactive visualisation and analysis of geological drilling data'
authors:
  - name: Gavin J Chapman
    orcid: 0009-0006-7734-3783
    affiliation: 1
affiliations:
 - name: Independent Researcher 
   index: 1
date: 8 July 2025 
bibliography: paper.bib
---

# Summary

GeoInsights 3D is an open-source Python application designed to accelerate the exploratory data analysis (EDA) workflow for geologists and data scientists working with drilling data. The software addresses the common bottleneck of data preparation, analysis, interpretation and modelling by providing an intuitive, web-based interface that transforms raw drilling data into interactive 3D models, statistical summaries, and machine learning insights in minutes. It enables users to visualise geological units, identify patterns in mineral concentrations, and test hypotheses dynamically, bridging the critical gap between raw data collection and advanced geological interpretation.

# Statement of Need

In the geological sciences, particularly within mineral exploration and resource geology, the standard workflow involves a progression from raw data to sophisticated 3D models. While powerful commercial software packages exist for detailed 3D geological modeling (e.g., Leapfrog, Vulcan) and advanced multivariate statistics (e.g., IoGAS), there is a distinct gap for a lightweight, accessible, and open-source tool dedicated to rapid first-pass analysis and automated modelling workflows. GeoInsights 3D is designed to fill this niche, serving as the crucial first step in the EDA process.

Built with a modern Python stack including Streamlit [@streamlit] for the user interface and Plotly [@plotly] for interactive visualisations, the application's backend is powered by a suite of robust open-source libraries. Data manipulation is handled by Pandas [@pandas_reback2020; @mckinney2010data], which relies on the foundational numerical computing capabilities of NumPy [@numpy_harris2020]. The scientific and machine learning features leverage Scikit-learn [@scikit-learn] for classical algorithms, SciPy [@scipy_virtanen2020] for scientific functions like interpolation, and TensorFlow [@tensorflow2015-whitepaper] for neural network models, with Matplotlib [@matplotlib_hunter2007] often supporting underlying plotting tasks. Furthermore, a key innovation of GeoInsights 3D is the integration of Large Language Models (LLMs) for geological interpretation and querying, a feature not yet standard in existing commercial or open-source geoscience software packages.

## Implementation and Core Features

The  contribution of GeoInsights 3D is embodied in its integrated suite of features, which are designed to address specific challenges in the geological workflow:

*   **Data Ingestion and Preparation:** The software directly handles common drilling data formats and includes a length-weighted compositing tool to standardize sample intervals, a critical prerequisite for reliable statistical analysis and modeling.

*   **Statistical Analysis:** The application provides rapid generation of key statistical summaries, including descriptive statistics tables, histograms, and box plots. This allows for immediate quantitative assessment of data distributions and relationships between variables.

*   **Interactive Visualisation and Spatial Analysis:** The core of the application is a 3D viewer for drillhole data. Users can dynamically filter by grade or lithology and apply vertical exaggeration. The software also generates 2D cross-sections (drill fences) and swath plots, which are essential tools for identifying spatial trends and geological continuity.

*   **Automated 3D Modeling:** To overcome the time-consuming nature of manual interpretation, the software includes automated tools for creating 3D grade shells. It uses Principal Component Analysis (PCA) to determine the natural anisotropy of mineralisation, followed by Radial Basis Function (RBF) interpolation to create an objective 3D model. This same engine also generates 3D solid models from categorical data, such as logged lithologies or the outputs from geochemical clustering, allowing for the visualisation of discrete geological bodies.

*   **Geochemical Domain Modeling:** The software implements PCA and K-Means clustering to define geological domains objectively from geochemical data. This provides a repeatable method to group samples with similar properties, helping to overcome the inherent subjectivity of manual logging. The results can be validated spatially in 3D and statistically with integrated plots.

*   **Explainable Machine Learning (XAI):** To move beyond "black box" models, the application integrates SHAP (SHapley Additive exPlanations) [@lundberg2017]. This allows geologists to interpret machine learning results, providing clear insights into the key geochemical drivers that control mineralisation.

*   **Generative AI Integration:** The software leverages Google Large Language Models to offer two novel features. The **AI Geo Data Interpreter** provides automated, context-aware geological summaries based on the loaded data. The **AI Data Playground** offers a natural language interface, allowing users to perform complex data queries, analyses and plots without writing code.

*   **Anomaly Detection for Data Quality Control:** The application employs Autoencoder neural networks for anomaly detection. This serves a dual purpose: identifying novel geochemical signatures that may warrant further exploration, and acting as a powerful data quality control tool to flag potential logging errors or sample swaps.

# Acknowledgements

I acknowledge the open-source communities behind the core Python libraries that made this software possible, including Streamlit, Plotly, Pandas, Scikit-learn, and the broader SciPy ecosystem.

# References