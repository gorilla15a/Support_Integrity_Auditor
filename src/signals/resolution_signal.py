def resolution_to_severity(hours):

    if hours >= 58:
        return 0

    elif hours >= 27:
        return 1

    elif hours >= 11:
        return 2

    else:
        return 3