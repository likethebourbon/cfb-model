import numpy as np
import pandas as pd
import re
from itertools import chain


def import_csvs(year=2018, weeks=16, encoding='ISO-8859-1', pbp_cols):
    for week in range(1, weeks+1):
        df = pd.DataFrame(pd.read_csv(
            f'../data/raw/pbp/{year} Week {week}.csv', encoding=encoding), columns=pbp_cols)
        df.fillna({'clock.minutes': 0, 'clock.seconds': 0}, inplace=True)
        df['week'] = week
        df = df.sort_values(['id'])
        df['id'] = df['id'].floordiv(1000000000)
        data = data.append(df, ignore_index=True)
        # finish establishing the data object and returning it from the function


def get_pbp(pbp_cols=['id', 'week', 'offense', 'offense_conference', 'defense', 'defense_conference', 'offense_score', 'defense_score', 'drive_id', 'period', 'clock.minutes', 'clock.seconds', 'yard_line', 'down', 'distance', 'yards_gained', 'play_type', 'play_text'],
            matchup_cols=['id', 'home_team', 'away_team', 'neutral_site', 'conference_game'],
            export_missing=True):

    # Import Play-by-Play and Matchup data #

    # Initialize necessary DataFrames
    data = pd.DataFrame(columns=pbp_cols)
    locations = pd.DataFrame(columns=matchup_cols)

    # Load each week's PBP data

    # Import each week's matchup data and append to the locations DataFrame
    for week in range(1, weeks+1):
        df = pd.DataFrame(pd.read_csv(
            f'{path}/raw/matchups/{year} Week {week}.csv', encoding="ISO-8859-1"), columns=matchup_cols)
        locations = locations.append(df, ignore_index=True)

    # store the matchups without PBP data in a separate DataFrame, and drop them from the main DataFrame
    pbp_missing = data[data['week'].isna()].copy()
    data.drop(data[data['week'].isna()].index[:], inplace=True)

    # change the dtypes of certain columns back to int or bool from float or object
    data_types = {
        'id': 'int64', 'week': 'int64', 'offense_score': 'int64',
        'defense_score': 'int64', 'drive_id': 'int64', 'period': 'int64',
        'clock.minutes': 'int64', 'clock.seconds': 'int64', 'yard_line': 'int64',
        'down': 'int64', 'distance': 'int64', 'yards_gained': 'int64',
        'neutral_site': 'bool', 'conference_game': 'bool'
        }

    # set dtypes for easier analysis
    data = data.astype(data_types)
    pbp_missing = pbp_missing.astype({'id': 'int64'})

    # make columns that isolate the scores for each team
    for place in 'home', 'away':
        offscore = data.loc[(data['offense'] == data[f'{place}_team']), 'offense_score'].rename(f'{place}score')
        defscore = data.loc[(data['defense'] == data[f'{place}_team']), 'defense_score'].rename(f'{place}score')
        data[f'{place}_score'] = pd.concat([offscore, defscore]).sort_index()

    # export the missing games if desired
    if export_missing:
        pbp_missing.to_csv(f'{path}/interim/errors/missing_games.csv')

    return data
