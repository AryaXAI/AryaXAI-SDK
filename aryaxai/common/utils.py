from datetime import datetime

def parse_float(s):
    """parse float from string, return None if not possible

    :param s: string to parse
    :return: float or None
    """
    try:
        return float(s)
    except ValueError:
        return None

def parse_datetime(s, format='%Y-%m-%d %H:%M:%S'):
    """Parse datetime from string, return None if not possible

    :param s: string to parse
    :param format: format string for datetime parsing
    :return: datetime or None
    """
    try:
        return datetime.strptime(s, format)
    except ValueError:
        return None

def pretty_date(date: str) -> str:
    """return date in format dd-mm-YYYY HH:MM:SS

    :param date: str datetime
    :return: pretty datetime
    """
    try:
        datetime_obj = datetime.strptime(date, '%Y-%m-%dT%H:%M:%S.%f')
    except ValueError:
        try:
            datetime_obj = datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            print("Date format invalid.")

    return datetime_obj.strftime('%d-%m-%Y %H:%M:%S')