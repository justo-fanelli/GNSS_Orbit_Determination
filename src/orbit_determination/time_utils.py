from datetime import datetime,timedelta

def datetime_to_julian(date: datetime) -> float:
    """Converts gregorian date `Timedate` object to Julian date.

    Parameters
    ----------
    date : Timedate
        Date to convert

    Returns
    -------
    float
        Julian date
    """
    if date.month == 1 or date.month == 2:
        Y = date.year - 1
        M = date.month + 12
    else:
        Y = date.year 
        M = date.month
    
    D = date.day + date.hour/24 + date.minute/1440 + (date.second+1e-6*date.microsecond)/86400
    A = int(Y/100)
    B = 2 - A + int(A/4)
    return int(365.25*(Y+4716)) + int((30.6001*(M+1))) + D + B - 1524.5

def julian_to_datetime(julian_date: float) -> datetime:
    """Converts a Julian Date to a UTC datetime object."""
    # Reference point: November 17, 1858, 00:00:00 UTC (MJD 0)
    # This corresponds to Julian Date 2400000.5
    reference_datetime = datetime(1858, 11, 17, 0, 0, 0)
    reference_julian_date = 2400000.5

    # Calculate the difference in days
    days_difference = julian_date - reference_julian_date

    # Add the difference to the reference datetime
    converted_datetime = reference_datetime + timedelta(days=days_difference)
    return converted_datetime
