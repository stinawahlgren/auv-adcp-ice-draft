# Mapping the ice draft of Dotson Ice Shelf with a long-range ADCP on an autonomous underwater vehicle
*Authors: S.Wahlgren, A. Wåhlin, K. J. Heywood*

This is a repository for the data analysis performed in  Wahlgren, S., Wåhlin, A. & Heywood, K. J (2026). Mapping the ice draft of Dotson Ice Shelf with a long-range ADCP on an autonomous underwater vehicle. *Journal of Atmospheric and Oceanic Technology*. [in review]

## Summary
A method is presented whereby the echo intensity from a long-range ADCP, mounted on an autonomous vehicle moving in a seabed-following mission beneath an Antarctic ice shelf, is used to produce a high resolution map of the ice base. The method provides a new data stream for ice base morphology, and can be obtained at lower risk compared to ice-following missions.  

## Workflow
The workflow is divided into two jupyter notebooks:
- 01_preprocess_echo_intensity.ipynb
- 02_derive_ice_draft.ipynb

In addition, the code for generating Figure 2-7 in Wahlgren et al. (2026) is provided.


## Data structure and availability
The data folder is divided into the following three subfolders:

- input : containing the ADCP echo intensities
- derived : where the generated datasets (ADCP ice draft and processed ADCP echo intensities) will be stored
- auxiliary  : containing CTD-, multibeam- and SAR datasets used for validation and sound speed corrections

The datasets required for the analysis are not provided here (except for SAR imagery over Dotson ice shelf), but are all publicly available and can be downloaded from repositories as specified below.

### data/input ### 
ADCP echo intensity can be downloaded from https://doi.org/10.5878/w1mp-x897 The following files are used:
  - NBP2202_02_ADCP_echo_intensity.nc
  - NBP2202_03_ADCP_echo_intensity.nc
  - NBP2202_04_ADCP_echo_intensity.nc


### data/auxiliary

- **data/auxiliary/multibeam:** Multibeam-derived ice draft maps can be downloaded from https://doi.org/10.5878/349w-b176 The following files are used:
  - NBP2202_09_10m.txt
  - NBP2202_14_10m.txt

- **data/auxiliary/CTD:** CTD data from  Hugin AUV missions in the Dotson Ice Shelf region during the NBP2202 Amundsen Sea expedition can be downloaded from https://doi.org/10.5878/349w-b176 The following AUV mission are from the Dotson Ice Shelf region:
  - NBP2202_002
  - NBP2202_003
  - NBP2202_004
  - NBP2202_005
  - NBP2202_006
  - NBP2202_007
  - NBP2202_008
  - NBP2202_009
  - NBP2202_010
  - NBP2202_011
  - NBP2202_014

- **data/auxiliary/background:** The provided SAR imagery over Dotson ice shelf was retrieved from Copernicus Browser https://browser.dataspace.copernicus.eu/ 2024-12-12.