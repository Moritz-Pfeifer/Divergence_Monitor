[![Paper Page](https://img.shields.io/badge/Paper_Page-SSRN%20preprint-green.svg)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5238187) [![Website](https://img.shields.io/badge/Website-Live%20Monitor-blue.svg)](https://moritz-pfeifer.github.io/eurozone-divergence-monitor/)


# Eurozone Divergence Monitor

This repository contains the code and data for the working paper:  
Bugdalle, T., Pfeifer, M. (2025). [“Warpings in time: Business and financial cycle synchronization in the euro area.”](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5238187). SSRN preprint. 

It provides scripts for measuring, analyzing, and visualizing the degree of business and financial cycle synchronization across euro area member states.

**Website:** [Eurozone Divergence Monitor](https://moritz-pfeifer.github.io/eurozone-divergence-monitor/)

---

## Repository Structure

- **`Data_final/`**  
  Includes precomputed data files required for the analysis.

- **`Scripts_Final/`**  
  Contains the final scripts and notebooks. 
  - **`Master_File_Cycles_29102024.ipynb`**: notebook for cycle measurement. It constructs the business and financial cycle indices using the specified data and methods.
  - **`Master_File_DTW.ipynb`**: notebook for distance measurement. 

- **`streamlit_app.py`**  
  A Streamlit application that provides the basis for the website implementation.

