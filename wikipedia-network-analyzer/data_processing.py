from datetime import datetime

def parse_date(date_str):
    '''Parse a date in the string format "YYYY-MM-DD HH:MM:SS" into a datetime object.
    Example: '2004-05-24 19:07:47' -> datetime(2004, 5, 24, 19, 7, 47)
    '''
    # Get the year, month, day, hour, minute, and second from the date string
    year, month, day = map(int, date_str[:10].split('-'))
    hour, minute, second = map(int, date_str[11:].split(':'))
    # Return a datetime object with manual inputs
    return datetime(year, month, day, hour, minute, second)