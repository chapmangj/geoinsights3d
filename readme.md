# GeoInsights 3D

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

GeoInsights 3D is an open-source Python application designed to accelerate the exploratory data analysis (EDA) workflow for geologists and data scientists working with drilling data.

## 📖 Tutorial & Documentation

A comprehensive, step-by-step tutorial is available that walks through all the major features of the application, from loading data to generating 3D models and running AI analysis.

**➡️ [Read the full tutorial on Medium](https://medium.com/@gavinjchapman/geoinsights-3d-rapid-drilling-data-visualisation-and-analysis-for-geologists-with-ai-1163b4039bb1)**

**Project Link: https://github.com/chapmangj/geoinsights3d**

**User Guide: https://github.com/chapmangj/geoinsights3d/blob/main/GeoInsights%203D%20User%20Guide.pdf**

## Key Features

*   **Data Ingestion and Preparation:** The software directly handles common drilling data formats and includes a length-weighted compositing tool to standardize sample intervals, a critical prerequisite for reliable statistical analysis and modeling.

*   **Statistical Analysis:** The application provides rapid generation of key statistical summaries, including descriptive statistics tables, histograms, and box plots. This allows for immediate quantitative assessment of data distributions and relationships between variables.

*   **Interactive Visualisation and Spatial Analysis:** The core of the application is a 3D viewer for drillhole data. Users can dynamically filter by grade or lithology and apply vertical exaggeration. The software also generates 2D cross-sections (drill fences) and swath plots, which are essential tools for identifying spatial trends and geological continuity.

*   **Automated 3D Modeling:** To overcome the time-consuming nature of manual interpretation, the software includes automated tools for creating 3D grade shells. It uses Principal Component Analysis (PCA) to determine the natural anisotropy of mineralisation, followed by Radial Basis Function (RBF) interpolation to create an objective 3D model. This same engine also generates 3D solid models from categorical data, such as logged lithologies or the outputs from geochemical clustering, allowing for the visualisation of discrete geological bodies.

*   **Geochemical Domain Modeling:** The software implements PCA and K-Means clustering to define geological domains objectively from geochemical data. This provides a repeatable method to group samples with similar properties, helping to overcome the inherent subjectivity of manual logging. The results can be validated spatially in 3D and statistically with integrated plots.

*   **Explainable Machine Learning (XAI):** To move beyond "black box" models, the application integrates SHAP (SHapley Additive exPlanations) [@lundberg2017]. This allows geologists to interpret machine learning results, providing clear insights into the key geochemical drivers that control mineralisation.

*   **Generative AI Integration:** The software leverages Google Large Language Models to offer two novel features. The **AI Geo Data Interpreter** provides automated, context-aware geological summaries based on the loaded data. The **AI Data Playground** offers a natural language interface, allowing users to perform complex data queries, analyses and plots without writing code.

*   **Anomaly Detection for Data Quality Control:** The application employs Autoencoder neural networks for anomaly detection. This serves a dual purpose: identifying novel geochemical signatures that may warrant further exploration, and acting as a powerful data quality control tool to flag potential logging errors or sample swaps.


## Installation

Using `pip`, you can install GeoInsights 3D directly from the Python Package Index (PyPI).

**1. Create and activate a virtual environment (Strongly Recommended):**

It is best practice to install packages in a dedicated virtual environment to avoid conflicts with other projects.

*   **For Windows:**
    ```powershell
    python -m venv venv
    .\venv\Scripts\activate
    ```

*   **For macOS/Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

**2. Install GeoInsights 3D:**

With your virtual environment active, install the package using `pip`:

```bash
pip install geoinsights-3d
```


**3. Set up API Key (Optional for AI features):**

The generative AI features require a Google Gemini API key.

Create a file named `.streamlit/secrets.toml` in the root of the project directory.

Add your API key to this file as follows:

.streamlit/secrets.toml
GOOGLE_API_KEY = "YOUR_API_KEY_HERE"


## Running the Application

Once the installation is complete, run the following command from the project's root directory:

geoinsights-3d
