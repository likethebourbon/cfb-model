import numpy as np
import pandas as pd
import re
from itertools import chain


def import_csvs(path='cfb-model/data/raw/', weeks=16, year=2018,
    pbp_cols=['id', 'week', 'offense', 'offense_conference', 
    'defense', 'defense_conference', 'offense_score', 'defense_score', 
    'drive_id', 'period', 'clock.minutes', 'clock.seconds', 'yard_line', 
    'down', 'distance', 'yards_gained', 'play_type', 'play_text'], 
    matchup_cols=['id', 'home_team', 'away_team', 'neutral_site', 'conference_game'],
    export_missing=True):
    
    ### Import Play-by-Play and Matchup data ###

    # Initialize necessary DataFrames
    data = pd.DataFrame(columns=pbp_cols)
    locations = pd.DataFrame(columns=matchup_cols)

    # Load each week's PBP data
    for week in range(1, weeks+1):
        df = pd.DataFrame(pd.read_csv(
            f'{path}/pbp/{year} Week {week}.csv', encoding="ISO-8859-1"), columns=pbp_cols)
        df.fillna({'clock.minutes': 0, 'clock.seconds': 0}, inplace=True)
        df['week'] = week
        df = df.sort_values(['id'])
        df['id'] = df['id'].floordiv(1000000000)
        data = data.append(df, ignore_index=True)

    # Import each week's matchup data and append to the locations DataFrame
    for week in range(1, weeks+1):
        df = pd.DataFrame(pd.read_csv(
            f'{path}/matchups/{year} Week {week}.csv', encoding="ISO-8859-1"), columns=matchup_cols)
        locations = locations.append(df, ignore_index=True)

    # store the matchups without PBP data in a separate DataFrame, 
    # and drop them from the main DataFrame
    pbp_missing = data[data['week'].isna()].copy()
    data.drop(data[data['week'].isna()].index[:], inplace=True)
