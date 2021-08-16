def heaviside_function(a):
    if a > 0:
        return 1
    elif a < 0:
        return 0
    elif a == 0:
        print("Warning! H(0)")
        return 0.5
