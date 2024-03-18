def print_dict(dct):
    for k, v in dct.items():
        if isinstance(v, float):
            print("%s: %.6f" % (k, v))
        else:
            print(f"{k}: {v}")