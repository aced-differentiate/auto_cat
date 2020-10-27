import os
from ase.build import bulk
from ase.data import atomic_numbers, ground_state_magnetic_moments, reference_states


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
            if species_list[i] in ["Fe", "Co", "Ni"]:  # check if ferromagnetic material
                bulk_obj.set_initial_magnetic_moments(
                    [ground_state_magnetic_moments[atomic_numbers[species_list[i]]]]
                    * len(bulk_obj)
                )
            bulk_obj.write("{}_bulk.i.traj".format(species_list[i]))
            os.chdir(curr_dir)
        i += 1
    print("\nCompleted")
