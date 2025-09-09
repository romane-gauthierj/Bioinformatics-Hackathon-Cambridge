from sklearn.preprocessing import MinMaxScaler

# ensure same patients IDS
# normalisation with min-max to ensure equal importance for each dataset
# remove missing data


def min_max_normalization(df):
    scaler = MinMaxScaler()
    normalized_df = (df.T - df.T.min()) / (df.T.max() - df.T.min())
    normalized_df = normalized_df.T
    return normalized_df


def remove_missing_values(df):
    df_clean = df.dropna(axis=0, how='any')
    return df_clean


def preprocess(df, common_cols):
    normalized_df = min_max_normalization(df)
    df_processed = remove_missing_values(normalized_df)
    df_processed = df_processed.reindex(columns=common_cols)
    return df_processed
