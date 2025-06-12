import math

def find_seniority(time, user_edits):
    """
    Takes input list of user edits by time and returns seniority measure of user at given time.
    Uses binary search algorithm.
    Inputs:
    - time: datetime object of time of revert
    - user_edits: list of dictionaries with keys 'time' and 'cum_edits', where index equals chronological order, represents one user's edits
    Output:
    - seniority measure (float)
    """
    # Create parameters for binary search
    low, high = 0, len(user_edits) - 1

    # Initialize output cumalative edits
    cumulative_edits = 0

    # Check if revert time is before user's first edit - this would be an error in the data
    if time < user_edits[0]['time']:
        # If so, return error a user cannot be reverted before their first edit
        return ValueError("Revert time is before reverted user's first edit")
    
    # Check if revert time is after last edit
    elif time >= user_edits[-1]['time']:
        # If so, return cumulative edits at last edit
        cumulative_edits = user_edits[-1]['cum_edits']
    
    # If revert time is between first and last edit
    else:
        # Run binary search to find closest time, note list is pre-indexed by time
        # Compare high-low values to find closest time
        while low <= high:
            # Use mid index to compare both upper and lower parts of search space
            mid = (low + high) // 2
            mid_time = user_edits[mid]['time']
            if mid_time == time:
                # If middle time is exact match, return cumulative edits at that time
                cumulative_edits = user_edits[mid]['cum_edits']
                return math.log10(cumulative_edits)
            # Otherwise, update search space based on comparison, and continue search
            elif mid_time < time:
                low = mid + 1
            else:
                high = mid - 1
        # Get the cumulative edits just before the specified time for seniority input
        closest_index = max(0, high)
        cumulative_edits = user_edits[closest_index]['cum_edits']
    # Return seniority measure
    return math.log10(cumulative_edits)