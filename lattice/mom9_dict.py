momDict = {
    0: "0 0 -3",
    1: "-1 -2 -2",
    2: "0 -2 -2",
    3: "1 -2 -2",
    4: "-2 -1 -2",
    5: "-1 -1 -2",
    6: "0 -1 -2",
    7: "1 -1 -2",
    8: "2 -1 -2",
    9: "-2 0 -2",
    10: "-1 0 -2",
    11: "0 0 -2",
    12: "1 0 -2",
    13: "2 0 -2",
    14: "-2 1 -2",
    15: "-1 1 -2",
    16: "0 1 -2",
    17: "1 1 -2",
    18: "2 1 -2",
    19: "-1 2 -2",
    20: "0 2 -2",
    21: "1 2 -2",
    22: "-2 -2 -1",
    23: "-1 -2 -1",
    24: "0 -2 -1",
    25: "1 -2 -1",
    26: "2 -2 -1",
    27: "-2 -1 -1",
    28: "-1 -1 -1",
    29: "0 -1 -1",
    30: "1 -1 -1",
    31: "2 -1 -1",
    32: "-2 0 -1",
    33: "-1 0 -1",
    34: "0 0 -1",
    35: "1 0 -1",
    36: "2 0 -1",
    37: "-2 1 -1",
    38: "-1 1 -1",
    39: "0 1 -1",
    40: "1 1 -1",
    41: "2 1 -1",
    42: "-2 2 -1",
    43: "-1 2 -1",
    44: "0 2 -1",
    45: "1 2 -1",
    46: "2 2 -1",
    47: "0 -3 0",
    48: "-2 -2 0",
    49: "-1 -2 0",
    50: "0 -2 0",
    51: "1 -2 0",
    52: "2 -2 0",
    53: "-2 -1 0",
    54: "-1 -1 0",
    55: "0 -1 0",
    56: "1 -1 0",
    57: "2 -1 0",
    58: "-3 0 0",
    59: "-2 0 0",
    60: "-1 0 0",
    61: "0 0 0",
    62: "1 0 0",
    63: "2 0 0",
    64: "3 0 0",
    65: "-2 1 0",
    66: "-1 1 0",
    67: "0 1 0",
    68: "1 1 0",
    69: "2 1 0",
    70: "-2 2 0",
    71: "-1 2 0",
    72: "0 2 0",
    73: "1 2 0",
    74: "2 2 0",
    75: "0 3 0",
    76: "-2 -2 1",
    77: "-1 -2 1",
    78: "0 -2 1",
    79: "1 -2 1",
    80: "2 -2 1",
    81: "-2 -1 1",
    82: "-1 -1 1",
    83: "0 -1 1",
    84: "1 -1 1",
    85: "2 -1 1",
    86: "-2 0 1",
    87: "-1 0 1",
    88: "0 0 1",
    89: "1 0 1",
    90: "2 0 1",
    91: "-2 1 1",
    92: "-1 1 1",
    93: "0 1 1",
    94: "1 1 1",
    95: "2 1 1",
    96: "-2 2 1",
    97: "-1 2 1",
    98: "0 2 1",
    99: "1 2 1",
    100: "2 2 1",
    101: "-1 -2 2",
    102: "0 -2 2",
    103: "1 -2 2",
    104: "-2 -1 2",
    105: "-1 -1 2",
    106: "0 -1 2",
    107: "1 -1 2",
    108: "2 -1 2",
    109: "-2 0 2",
    110: "-1 0 2",
    111: "0 0 2",
    112: "1 0 2",
    113: "2 0 2",
    114: "-2 1 2",
    115: "-1 1 2",
    116: "0 1 2",
    117: "1 1 2",
    118: "2 1 2",
    119: "-1 2 2",
    120: "0 2 2",
    121: "1 2 2",
    122: "0 0 3",
}

# {
#   0: "0 0 0",
#   1: "0 0 1",
#   2: "0 0 -1",
#   3: "0 1 0",
#   4: "0 -1 0",
#   5: "1 0 0",
#   6: "-1 0 0",
#   7: "0 1 1",
#   8: "0 1 -1",
#   9: "0 -1 1",
#   10: "0 -1 -1",
#   11: "1 0 1",
#   12: "1 0 -1",
#   13: "-1 0 1",
#   14: "-1 0 -1",
#   15: "1 1 0",
#   16: "1 -1 0",
#   17: "-1 1 0",
#   18: "-1 -1 0",
#   19: "1 1 1",
#   20: "1 1 -1",
#   21: "1 -1 1",
#   22: "1 -1 -1",
#   23: "-1 1 1",
#   24: "-1 1 -1",
#   25: "-1 -1 1",
#   26: "-1 -1 -1",
# }


def dictToList():
    momList = []
    for key, val in momDict.items():
        momList.append(tuple([int(p) for p in val.split(" ")]))
    return momList
