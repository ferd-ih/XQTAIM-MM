# ss_qtaim_gen
This directory contains scripts for generating, processing, and analyzing QTAIM-based graph representations of molecular crystals starting from electron density analysis files. The scripts extract critical point (CP) properties, compute graph descriptors, and perform feature embedding for machine learning applications.

## Directory structure

. \
├── parse_qtaim_fi.py        # Utils script: Parses QTAIM output files and extracts relevant data \
├── get_cp_fts_fi.py         # Extracts QTAIM CPs & descriptors from CPprop.txt and store as jsons \
├── gen_qtaim_graph.py       # Generates QTAIM-based molecular graphs as gml files \