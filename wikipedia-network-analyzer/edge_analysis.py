from datetime import timedelta

def get_ab_ba_sequences(edge_list):
    """
    Find AB-BA sequences in a list of network edges. Where one user reverts another, and 
    then the reverted user reverts the original reverter within 24 hours.
    Takes input list of network edges and returns list of edges (dictionaries) corresponding to the BA revert. 
    Inputs:
    - edge_list: list of dictionaries with keys 'reverter', 'reverted', 'time', 'reverter_seniority', 'reverted_seniority'
    Output:
    - ab_ba_sequences: list of list of dictionaries with each list corresponding to an AB-BA sequence
    """
    # Sort the edge list by time
    edge_list.sort(key=lambda x: x['time'])

    # Track the last revert times
    last_revert_info = {}

    # Store the AB-BA sequences
    ab_ba_sequences = []

    # Iterate through the edge list
    for edge in edge_list:
        reverter, reverted, time = edge['reverter'], edge['reverted'], edge['time']
        
        # Check for BA revert after AB revert in dictionary
        if (reverted, reverter) in last_revert_info:
            # Check if the BA revert is also within 24 hours of the AB revert
            if time - last_revert_info[(reverted, reverter)]['time'] <= timedelta(hours=24):
                # If so, store the BA revert row in list of [AB, BA] edges
                ab_ba_sequences.append([last_revert_info[(reverted, reverter)], edge])

        # Update the last revert time for AB
        last_revert_info[(reverter, reverted)] = edge

    return ab_ba_sequences