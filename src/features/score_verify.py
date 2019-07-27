import pandas as pd
import numpy as np
from itertools import chain


def get_drive_counts(side, pbp, drives):
    """Return the difference in the offense's drive counts between the two
    datasets for all games of a specified side.

    The first step in verifying the scores by recreating each drive of the
    game is making sure that the play-by-play and drives datasets have the same
    number of drives for each side in each game. This function returns the
    number of drives in pbp dataset minus the number of drives in the drives
    dataset for a given game_id. The ideal result is 0 for each game.

    The pbp dataset begins drives with the kicking team, resulting in one-play
    drives for the kicking team. This isn't how the drives dataset counts
    drives, so we must exclude plays that contain the word 'kickoff' before
    counting the number of drives.

    Arguments:
    side -- a str of either 'home' or 'away'
    pbp -- name of the play-by-play DataFrame
    drives -- name of the drives DataFrame
    """
    pbp_drives = (
        pbp[~(pbp["play_type"].str.contains("Kickoff"))]
        .loc[pbp["offense"] == pbp[f"{side}_team"]]
        .groupby("id")
        .agg({"drive_id": "nunique"})
    )
    pbp_drives.rename({"drive_id": "pbp_drives"}, axis=1, inplace=True)
    drive_chart_drives = (
        drives.loc[drives["offense"] == drives[f"{side}_team"]]
        .groupby("id")
        .agg({"drive_id": "nunique"})
    )
    drive_chart_drives.rename({"drive_id": "drive_chart_drives"}, axis=1, inplace=True)
    df = pbp_drives.join(drive_chart_drives)
    df["difference"] = df["pbp_drives"] - df["drive_chart_drives"]
    df["side"] = side
    return df


def import_drives(year, weeks):
    drives = pd.DataFrame()
    for week in range(1, weeks + 1):
        df = pd.DataFrame(
            pd.read_csv(
                f"../data/raw/drives/{year} Week {week}.csv", encoding="ISO-8859-1"
            )
        )
        df["week"] = week
        drives = drives.append(df, ignore_index=True, sort=False)
    drives = drives.rename({"id": "drive_id", "game_id": "id"}, axis=1)

    loc_columns = ["id", "home_team", "away_team", "neutral_site", "conference_game"]
    locations = pd.DataFrame(columns=loc_columns)

    for week in range(1, weeks + 1):
        df = pd.DataFrame(
            pd.read_csv(
                f"../data/raw/matchups/{year} Week {week}.csv", encoding="ISO-8859-1"
            ),
            columns=loc_columns,
        )
        locations = locations.append(df, ignore_index=True)

    locations = locations.sort_values(by="id").reset_index(drop=True)
    drives = pd.merge(drives, locations, on="id", how="left")
    return drives


def get_pbp_drives(df, side):
    drives = (
        df[
            (df["offense"] == df[f"{side}_team"])
            & (~(df["play_type"].str.contains("Kickoff")))
        ]
        .groupby("id")
        .agg({"drive_id": "unique"})
        .rename({"drive_id": "pbp_drive_id"}, axis=1)
    )
    return drives


def get_dc_drives(df, side):
    drives = (
        df[df["offense"] == df[f"{side}_team"]]
        .groupby("id")
        .agg({"drive_id": "unique"})
        .rename({"drive_id": f"dc_drive_id"}, axis=1)
    )
    return drives


def get_missing_drives(pbp, drives):
    home_drive_counts = (
        get_drive_counts("home", pbp, drives)
        .join(get_pbp_drives(pbp, "home"))
        .join(get_dc_drives(drives, "home"))
    )
    away_drive_counts = (
        get_drive_counts("away", pbp, drives)
        .join(get_pbp_drives(pbp, "away"))
        .join(get_dc_drives(drives, "away"))
    )
    all_drive_counts = home_drive_counts.append(away_drive_counts)
    all_drive_counts["extra_pbp_drives"] = all_drive_counts.apply(
        lambda x: np.setdiff1d(x.pbp_drive_id, x.dc_drive_id), axis=1
    )
    all_drive_counts["extra_pbp_drives"] = all_drive_counts["extra_pbp_drives"].apply(
        lambda x: np.nan if len(x) == 0 else x
    )
    all_drive_counts["extra_dc_drives"] = all_drive_counts.apply(
        lambda x: np.setdiff1d(x.dc_drive_id, x.pbp_drive_id), axis=1
    )
    all_drive_counts["extra_dc_drives"] = all_drive_counts["extra_dc_drives"].apply(
        lambda x: np.nan if len(x) == 0 else x
    )
    return all_drive_counts


def get_invalid_score_changes(pbp):
    """Return a DataFrame of plays with invalid score changes, and the plays immediately
    before and after.

    Valid score changes are 2, 3, 6, 7, and 8 points. There are occasional
    one-point plays, but they are exceedingly rare. 

    Arguments:
    pbp -- name of the play-by-play DataFrame
    """
    ends_of_games = [
        pbp.loc[pbp["id"] == game].index[-1] for game in pbp["id"].unique()
    ]
    home_scores_to_check = pbp.loc[
        ~(pbp.home_score.diff().isin([0, 2, 3, 6, 7, 8])) & (pbp.home_score != 0)
    ]
    away_scores_to_check = pbp.loc[
        ~(pbp.away_score.diff().isin([0, 2, 3, 6, 7, 8])) & (pbp.away_score != 0)
    ]
    scores_to_check = home_scores_to_check.append(away_scores_to_check)
    scores_to_check = scores_to_check.sort_index()
    scores_index = [
        item
        for item in scores_to_check.index.tolist()
        if item - 1 not in ends_of_games and item not in ends_of_games
    ]
    return pbp.loc[
        list(
            chain(
                *zip(
                    [item - 1 for item in scores_index],
                    scores_index,
                    [item + 1 for item in scores_index],
                    [item + 2 for item in scores_index],
                    [item + 3 for item in scores_index],
                )
            )
        )
    ]


def fix_scores(pbp, num_plays=5):
    """Using the DataFrame returned by get_invalid_score_changes(), change the invalid
    scores in the play-by-play DataFrame to the mode of the scores in each group of
    three plays.

    The most common invalid score issue is a penalty occuring on the scoring play,
    leading to the score not registering on the next play and being reinstating
    on the play after that. This function sets the scores for both sides to the mode
    of the scores on these three plays.

    Arguments:
    pbp -- the name of the play-by-play DataFrame
    """
    scores_to_fix = get_invalid_score_changes(pbp)
    home_modes = scores_to_fix.groupby(np.arange(len(scores_to_fix)) // num_plays)[
        "home_score"
    ].apply(lambda x: x.mode()[0])
    away_modes = scores_to_fix.groupby(np.arange(len(scores_to_fix)) // num_plays)[
        "away_score"
    ].apply(lambda x: x.mode()[0])
    for n in range(len(home_modes)):
        pbp.loc[
            scores_to_fix.iloc[
                n * num_plays : (n * num_plays) + num_plays
            ].index.tolist(),
            "home_score",
        ] = home_modes[n]
        pbp.loc[
            scores_to_fix.iloc[
                n * num_plays : (n * num_plays) + num_plays
            ].index.tolist(),
            "away_score",
        ] = away_modes[n]
    return pbp


def import_matchups(year, weeks):
    matchup_df = pd.DataFrame()
    for week in range(1, weeks + 1):
        df = pd.DataFrame(
            pd.read_csv(
                f"../data/raw/matchups/{year} Week {week}.csv", encoding="ISO-8859-1"
            )
        )
        matchup_df = matchup_df.append(df, ignore_index=True, sort=False)
    matchup_df["start_date"] = pd.to_datetime(matchup_df["start_date"])
    return matchup_df


def compare_pbp_matchup(pbp, matchup):
    """Get the differences in scores between play-by-play and matchup DataFrames.

    Arguments:
    pbp -- name of the play-by-play DataFrame
    matchup -- name of the matchup DataFrame
    """
    periods = [
        pbp.loc[(pbp["id"] == game) & (pbp["period"] == period)].index[-1]
        for game in pbp["id"].unique()
        for period in pbp.loc[pbp["id"] == game, "period"].unique()
    ]
    pbp_scores = (
        pbp[
            [
                "id",
                "week",
                "home_team",
                "away_team",
                "period",
                "home_score",
                "away_score",
            ]
        ]
        .loc[periods]
        .copy()
        .pivot_table(index=["id", "week", "home_team", "away_team"], columns="period")
    )
    pbp_scores.columns = [
        f"{pbp_scores.columns.values[i][0]}_q{str(pbp_scores.columns.values[i][1])}"
        for i in range(len(pbp_scores.columns.values))
    ]
    pbp_scores = (
        pbp_scores[
            [
                *[f"home_score_q{i}" for i in range(1, 6)],
                *[f"away_score_q{i}" for i in range(1, 6)],
            ]
        ]
        .reset_index()
        .set_index("id", drop=True)
        .sort_index()
    )
    matchup_scores = (
        matchup[
            [
                "id",
                "week",
                "home_team",
                "away_team",
                *[f"home_line_scores[{i}]" for i in range(5)],
                *[f"away_line_scores[{i}]" for i in range(5)],
            ]
        ]
        .set_index("id", drop=True)
        .sort_index()
    )
    matchup_scores.columns = [
        "week",
        "home_team",
        "away_team",
        *[f"home_score_q{i}" for i in range(1, 6)],
        *[f"away_score_q{i}" for i in range(1, 6)],
    ]
    score_diffs = pbp_scores[["week", "home_team", "away_team"]].copy()
    for side in "home", "away":
        cum_matchup_scores = matchup_scores[
            [f"{side}_score_q{i}" for i in range(1, 6)]
        ].cumsum(axis=1)
        for quarter in range(1, 6):
            score_diffs[f"{side}_score_q{quarter}"] = (
                pbp_scores[f"{side}_score_q{quarter}"]
                - cum_matchup_scores[f"{side}_score_q{quarter}"]
            )
    return score_diffs


def nonzero_score_diffs(pbp, matchup):
    """Get the differences in scores between play-by-play and matchup DataFrames
    that are not equal to zero in any period.

    Arguments:
    pbp -- name of the play-by-play DataFrame
    matchup -- name of the matchup DataFrame
    """
    diffs = compare_pbp_matchup(pbp, matchup)
    non_zero = diffs.loc[
        (diffs["home_score_q1"] != 0)
        | (diffs["home_score_q2"] != 0)
        | (diffs["home_score_q3"] != 0)
        | (diffs["home_score_q4"] != 0)
        | (diffs.fillna(0)["home_score_q5"] != 0)
        | (diffs["away_score_q1"] != 0)
        | (diffs["away_score_q2"] != 0)
        | (diffs["away_score_q3"] != 0)
        | (diffs["away_score_q4"] != 0)
        | (diffs.fillna(0)["away_score_q5"] != 0)
    ]
    return non_zero
