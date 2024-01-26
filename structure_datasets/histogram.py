import pandas as pd


def histogram(values, MAX, STEP):
    intervals, counts = [], []
    for left in range(0, MAX, STEP):
        count = 0
        for key, value in values.items():
            if left <= key < left + STEP:
                count += value
        intervals.append(f"{left}-{left + STEP - 1}")
        counts.append(count)

    count = 0
    for key, value in values.items():
        if MAX <= key:
            count += value
    intervals.append(f"≥{MAX}")
    counts.append(count)

    return pd.DataFrame(
        index=intervals,
        data={
            "num": counts
        }
    )
