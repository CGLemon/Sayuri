def is_float(val):
    try:
        float(val)
    except ValueError:
        return False
    return True

def is_int(val):
    try:
        int(val)
    except ValueError:
        return False
    return True
