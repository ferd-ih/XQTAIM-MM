import json
import os
import numpy as np


def parse_cp(lines, verbose=True):
    """
    Takes
        lines: list of lines from cp file
    Returns
        cp_dict: dictionary of critical point information
    """
    lines_split = [line.split() for line in lines]
    cp_bond, cp_atom = False, False
    cp_name = "null"
    cp_dict = {}
    if "(3,-3)" in lines_split[0]:
        cp_atom = True
        if verbose:
            print("atom cp")

    elif "(3,-1)" in lines_split[0]:
        cp_bond = True
        if verbose:
            print("bond cp")

    else:
        if verbose:
            print("ring critical bond not implemented")
        return "ring", cp_dict

    cp_atom_conditionals = {
        "cp_num": ["----------------"],
        "ele_info": ["Corresponding", "nucleus:"],
        "pos_ang": ["Position", "(Angstrom):"],
        "e_density": ["Density", "of", "all", "electrons:"],
        "lol": ["Localized", "orbital", "locator"],
        "energy_density": ["Energy", "density", "E(r)"],
        "Lagrangian_K": ["Lagrangian", "kinetic", "energy"],
        "Hamiltonian_K": ["Hamiltonian", "kinetic", "energy"],
        "Potential_E": ["Potential", "energy", "density"],
        "lap_e_density": ["Laplacian", "electron", "density:"],
        "e_loc_func": ["Electron", "localization", "function"],
        "ave_loc_ion_E": ["Average", "local", "ionization", "energy"],
        "delta_g_promolecular": ["Delta-g", "promolecular"],
        "delta_g_hirsh": ["Delta-g", "Hirshfeld"],
        "iri": ["Interaction", "region", "indicator"],
        "sign_lambda2_rho": ["Sign(lambda2)*rho"],
        # "rdg": ["Reduced", "density", "gradient"],
        "esp_nuc": ["ESP", "nuclear", "charges:"],
        "esp_e": ["ESP", "electrons:"],
        "esp_total": ["Total", "ESP:"],
        "grad_norm": ["Components", "gradient", "x/y/z"],
        "lap_norm": ["Components", "Laplacian", "x/y/z"],
        "eig_hess": ["Eigenvalues", "Hessian:"],
        "det_hessian": ["Determinant", "Hessian:"],
        "ellip_e_dens": ["Ellipticity", "electron", "density:"],
        "eta": ["eta", "index:"],
    }

    cp_bond_conditionals = {
        "cp_num": ["----------------"],
        "connected_bond_paths": ["Connected", "atoms:"],
        "pos_ang": ["Position", "(Angstrom):"],
        "e_density": ["Density", "of", "all", "electrons:"],
        "lol": ["Localized", "orbital", "locator"],
        "energy_density": ["Energy", "density", "E(r)"],
        "Lagrangian_K": ["Lagrangian", "kinetic", "energy"],
        "Hamiltonian_K": ["Hamiltonian", "kinetic", "energy"],
        "Potential_E": ["Potential", "energy", "density"],
        "lap_e_density": ["Laplacian", "electron", "density:"],
        "e_loc_func": ["Electron", "localization", "function"],
        "ave_loc_ion_E": ["Average", "local", "ionization", "energy"],
        "delta_g_promolecular": ["Delta-g", "promolecular"],
        "delta_g_hirsh": ["Delta-g", "Hirshfeld"],
        "iri": ["Interaction", "region", "indicator"],
        "sign_lambda2_rho": ["Sign(lambda2)*rho"],
        # "rdg": ["Reduced", "density", "gradient"],
        "esp_nuc": ["ESP", "nuclear", "charges:"],
        "esp_e": ["ESP", "electrons:"],
        "esp_total": ["Total", "ESP:"],
        "grad_norm": ["Components", "gradient", "x/y/z"],
        "lap_norm": ["Components", "Laplacian", "x/y/z"],
        "eig_hess": ["Eigenvalues", "Hessian:"],
        "det_hessian": ["Determinant", "Hessian:"],
        "ellip_e_dens": ["Ellipticity", "electron", "density:"],
        "eta": ["eta", "index:"],
    }

    if cp_atom:
        unknown_id = 0
        for ind, i in enumerate(lines_split):
            for k, v in cp_atom_conditionals.items():
                if all(x in i for x in v):
                    if k == "cp_num":
                        cp_dict[k] = int(i[2][:-1])

                    elif k == "pos_ang":
                        cp_dict[k] = [float(x) for x in i[2:]]

                    elif k == "ele_info":
                        if i[2] == "Unknown":
                            cp_name = str(unknown_id) + "_Unknown"
                            cp_dict["number"] = "Unknown"
                            cp_dict["ele"] = "Unknown"
                            unknown_id += 1
                        else:
                            if len(i) == 3:
                                cp_dict["element"] = i[2].split("(")[1][:-1]
                                cp_dict["number"] = i[2].split("(")[0]
                            else:
                                cp_dict["element"] = i[2].split("(")[1]
                                cp_dict["number"] = i[2].split("(")[0]
                            cp_name = cp_dict["number"] + "_" + cp_dict["element"]

                    elif k == "esp_total":
                        cp_dict[k] = float(i[2])

                    elif k == "eig_hess":
                        cp_dict[k] = np.sum(np.array([float(x) for x in i[-3:]]))

                    elif k == "grad_norm" or k == "lap_norm":
                        cp_dict[k] = float(lines_split[ind + 2][-1])

                    else:
                        # print(i)
                        cp_dict[k] = float(i[-1])

                    cp_atom_conditionals.pop(k)
                    break

    elif cp_bond:
        for ind, i in enumerate(lines_split):
            for k, v in cp_bond_conditionals.items():
                if all(x in i for x in v):
                    if k == "cp_num":
                        cp_dict[k] = int(i[2][:-1])
                        cp_name = str(cp_dict[k]) + "_bond"
                    elif k == "connected_bond_paths":
                        
                        list_raw = [x for x in i[2:]]
                        # save only items that contain a number
                        list_raw = [
                            x for x in list_raw if any(char.isdigit() for char in x)
                        ]
                        #print("list raw connected: ", list_raw)
                        # list_raw = [list_raw[0], list_raw[-2]]
                        list_raw = [int(x.split("(")[0]) for x in list_raw]
                        #print("list raw connected: ", list_raw)
                        cp_dict[k] = list_raw
                    elif k == "pos_ang":
                        cp_dict[k] = [float(x) for x in i[2:]]
                    elif k == "esp_total":
                        cp_dict[k] = float(i[2])
                    elif k == "eig_hess":
                        cp_dict[k] = np.sum(np.array([float(x) for x in i[-3:]]))
                    elif k == "grad_norm" or k == "lap_norm":
                        cp_dict[k] = float(lines_split[ind + 2][-1])

                    else:
                        cp_dict[k] = float(i[-1])
                    # print(v)
                    cp_bond_conditionals.pop(k)

                    break

    else:
        print("error - ring and cage critical points not implemented")

    return cp_name, cp_dict


def get_qtaim_descs(file="./CPprop_1157_1118_1158.txt", verbose=False):
    """
    helper function to parse CPprop file from multiwfn.
    Takes
        file (str): path to CPprop file
        verbose(bool): prints dictionary of descriptors
    returns: dictionary of descriptors
    """
    cp_dict, ret_dict = {}, {}

    with open(file) as f:
        lines = f.readlines()
        lines = [line[:-1] for line in lines]

    # section lines into segments on ----------------
    track = 0
    for ind, line in enumerate(lines):
        if "----------------" in line:
            lines_segment = []
        lines_segment.append(line)
        if ind < len(lines) - 1:
            if "----------------" in lines[ind + 1]:
                cp_dict[track] = lines_segment
                track += 1
        else:
            cp_dict[track] = lines_segment

    for k, v in cp_dict.items():
        ind_atom, cp_dict = parse_cp(v, verbose=verbose)
        ret_dict[ind_atom] = cp_dict

    # remove keys-value pairs that are "ring"
    ret_dict = {k: v for k, v in ret_dict.items() if "ring" not in k}
    return ret_dict


def only_atom_cps(qtaim_descs):
    """
    separates qtaim descriptors into atom and bond descriptors
    """
    ret_dict = {}
    ret_dict_bonds = {}
    for k, v in qtaim_descs.items():
        if "bond" not in k and "Unknown" not in k:
            ret_dict[k] = v
        if "bond" in k:
            ret_dict_bonds[k] = v
    return ret_dict, ret_dict_bonds
