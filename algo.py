import pandas as pd


def will_surpass_best(target, optimization_history, min_delta, min_steps=5, direction='min'):
    if len(optimization_history) < min_steps:  # not enough data
        return True

    smooth_window = 1
    complete = False
    while not complete:
        temp = list(pd.Series(optimization_history).rolling(smooth_window).mean())
        if len(temp) == 2:  # cannot be smoothed any further
            break
        needs_further_smoothing = False
        for x in range(len(temp) - 2):
            if (temp[x] < temp[x + 1] and temp[x + 1] > temp[x + 2]) or \
                    (temp[x] > temp[x + 1] and temp[x + 1] < temp[x + 2]):
                needs_further_smoothing = True

        if needs_further_smoothing:
            smooth_window += 1
        else:
            optimization_history = temp
            complete = True

    curr_val = optimization_history[-1]

    def get_dx(arr):
        return [arr[x + 1] - arr[x] for x in range(len(arr) - 1)]

    dx = []
    while len(optimization_history) > 1:  # calculate dx orders (0 index is first dx)
        optimization_history = get_dx(optimization_history)
        dx.append(optimization_history)

    dx = [d[-1] for d in dx]  # get most recent elem of each dx order

    while True:
        if curr_val >= target:  # has potential to reach best
            return True

        # get predicted slope of *next* step
        for x in reversed(range(len(dx) - 1)):
            dx[x] += dx[x + 1]

        # if step too small or negative slope
        if abs(dx[0]) < min_delta or (dx[0] < 0 if direction == 'max' else dx[0] > 0):
            return False

        # use predicted slope to take a step
        curr_val += dx[0]
