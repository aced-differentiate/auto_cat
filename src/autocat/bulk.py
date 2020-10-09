import os
from ase.build import bulk


def gen_bulk_dirs(species_list, a_dict={}, c_dict={}):
    """
    Generates bulk structures in separate directories

    Parameters:
        species_list(list of str): list of bulk species to be considered
        a_dict(dict): if lattice parameters want to be specified for a given species
        c_dict(dict): if lattice parameters want to be specified for a given species
    Returns:
        None
    """
    curr_dir = os.getcwd()
    i = 0
    print("\nStarted creating bulk directories")
    while i < len(species_list):

        # Check if a manually specified
        if species_list[i] in a_dict:
            a = a_dict[species_list[i]]
        else:
            a = None

        # Check if c manually specified
        if species_list[i] in c_dict:
            c = c_dict[species_list[i]]
        else:
            c = None

        try:
            os.mkdir(species_list[i] + "_bulk")
        except OSError:
            print("\nFailed Creating Directory ./{}".format(species_list[i] + "_bulk"))
        else:
            print(
                "\nSuccessfully Created Directory ./{}".format(
                    species_list[i] + "_bulk"
                )
            )
            os.chdir(species_list[i] + "_bulk")
            bulk_obj = bulk(species_list[i], a=a, c=c)
            bulk_obj.write("{}_bulk.traj".format(species_list[i]))
            os.chdir(curr_dir)
        i += 1
    print("\nCompleted")
