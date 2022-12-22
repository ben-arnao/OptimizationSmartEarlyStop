import numpy as np
import pandas as pd


def will_surpass_best(target, optimization_history, min_delta=1e-1, min_steps=5, direction='min'):
    if min_steps < 4 or len(optimization_history) < 4:
        raise Exception('need at least 4 data points')
    if len(optimization_history) < min_steps or pd.isna(target):  # not enough data
        return True

    # smooth the vector enough to properly analyze
    smooth_window = 1
    complete = False
    while not complete:
        temp = list(pd.Series(optimization_history).rolling(smooth_window).mean().dropna())
        if len(temp) == 2:  # cannot be smoothed any further
            smoothed_history = temp
            break
        needs_further_smoothing = False

        for x in range(len(temp) - 3):
            roc_one = temp[x + 1] - temp[x]
            roc_two = temp[x + 2] - temp[x + 1]
            roc_three = temp[x + 3] - temp[x + 2]

            # make sure that slope is going in the same direction among 3 slope points (4 data points)
            if (roc_one < roc_two and roc_two > roc_three) or \
                    (roc_one > roc_two and roc_two < roc_three):
                needs_further_smoothing = True

        if needs_further_smoothing:
            smooth_window += 1
        else:
            smoothed_history = temp
            complete = True

    x = [x for x in range(len(smoothed_history))]
    poly = np.polyfit(x, smoothed_history, deg=2)  # smoothed optimization curve needs 2nd degree
    curr_x = x[-1] + 1
    prev_y = None
    # now we step forward and simulate future optimization curve
    while True:
        curr_x += 1
        curr_y = np.polyval(poly, curr_x)
        if curr_y > target:  # simulated curve has passed target, return True
            return True
        if prev_y is not None:
            # stop simulating curve if step is too small, or slope starts going in wrong direction
            if abs(prev_y - curr_y) < min_delta or curr_y < prev_y if direction == 'max' else curr_y > prev_y:
                return False
        prev_y = curr_y
