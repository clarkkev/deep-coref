import time

starts = {}
totals = {}
counts = {}


def start(key):
    starts[key] = time.time()


def stop(key):
    if key not in totals:
        totals[key] = 0
        counts[key] = 0
    counts[key] += 1
    totals[key] += time.time() - starts[key]


def print_totals():
    print("Times:", ", ".join(["{:}: {:.1f}".format(key, totals[key]) for key in totals]))


def total_time(key):
    return totals[key]


def avg_time(key):
    return totals[key] / counts[key]


def clear():
    global starts, totals, counts
    starts, totals, counts = {}, {}, {}
