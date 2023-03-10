import os

import pandas as pd
from fire import Fire


def main(datadir="data"):
    df = pd.read_csv(datadir + "/best_tracks.csv", skiprows=[1, 2])
    for idx, group in df.groupby("Storm ID"):
        group = group[
            [
                "Central Pressure",
                "Forward Speed",
                "Heading",
                "Holland B1",
                "Radius Max Winds",
                "Radius Pressure 1",
                "Storm Latitude",
                "Storm Longitude",
            ]
        ]
        group.to_csv(datadir + f"/storms/s{int(idx):03}/best_track.csv", index=False)


if __name__ == "__main__":
    Fire(main)
