import math


# see http://stackoverflow.com/questions/3018758/determine-precision-and-scale-of-particular-number-in-python
def precision_and_scale(x):
    max_digits = 14
    int_part = int(abs(x))
    magnitude = 1 if int_part == 0 else int(math.log10(int_part)) + 1
    if magnitude >= max_digits:
        return magnitude, 0
    fractionary_part = abs(x) - int_part
    multiplier = 10 ** (max_digits - magnitude)
    fractionary_digits = multiplier + int(multiplier * fractionary_part + 0.5)
    while fractionary_digits % 10 == 0:
        fractionary_digits /= 10
    scale = int(math.log10(fractionary_digits))
    return magnitude + scale, scale
