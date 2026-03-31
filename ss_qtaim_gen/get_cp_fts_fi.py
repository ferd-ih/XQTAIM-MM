import os
import json
import numpy as np
from parse_qtaim_fi import (
    get_qtaim_descs,
    only_atom_cps
)

# root_test = "/ocean/projects/che250019p/fihiri/OMDB_data/omdb_PBE_CPProp"
# root_test = "/ocean/projects/che250019p/fihiri/OCELOT_data/ocelot_cpprop_pbe"
root_test = "/ocean/projects/che250019p/fihiri/ROY_data/ROY_pbe_CPProp"
for molecule_id in os.listdir(root_test):
    QTAIM_loc = os.path.join(root_test, molecule_id)

    if os.path.isdir(QTAIM_loc):
        cp_file = os.path.join(QTAIM_loc, "CPprop.txt")
        if os.path.exists(cp_file):
            try:
                results = {}
                df_cp = get_qtaim_descs(cp_file, verbose=False)
                atom_cp, bond_cp = only_atom_cps(df_cp)
                ncp_data = {molecule_id: atom_cp}
                bcp_data = {molecule_id: bond_cp}
                output_file_ncp = os.path.join(QTAIM_loc, f"{molecule_id}_qtaim_ncp.json")
                with open(output_file_ncp, 'w') as json_file:
                    json.dump(ncp_data, json_file, indent=4)
                output_file_bcp = os.path.join(QTAIM_loc, f"{molecule_id}_qtaim_bcp.json")
                with open(output_file_bcp, 'w') as json_file:
                    json.dump(bcp_data, json_file, indent=4)

                print(f"QTAIM descriptors for molecule {molecule_id} extracted and saved successfully.")

            except Exception as e:
                print(f"Error processing CPprop.txt for molecule {molecule_id}: {e}")
        else:
            print(f"CPprop.txt not found for molecule: {molecule_id}")
    else:
        print(f"No QTAIM folder for molecule: {molecule_id}")