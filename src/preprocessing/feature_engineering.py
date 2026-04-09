import pandas as pd


def replace_age_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Conversion of age ranges to numeric midpoints.
    """
    age_map = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25,
        '[30-40)': 35, '[40-50)': 45, '[50-60)': 55,
        '[60-70)': 65, '[70-80)': 75, '[80-90)': 85,
        '[90-100)': 95
    }
    df['age'] = df['age'].map(age_map)
    return df


def group_admission_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Grouping admission types.
    """
    df['admission_type_id'] = df['admission_type_id'].replace({
        1: 'Emergency',
        2: 'Urgent',
        3: 'Elective'
    })
    return df


def group_admission_source(df: pd.DataFrame) -> pd.DataFrame:
    """
    Grouping admission source IDs.
    """
    df['admission_source_id'] = df['admission_source_id'].replace({
        1: 'Referral',
        7: 'Emergency',
        9: 'Transfer'
    })
    return df


def diag_cluster(diag):
    """
    Cluster diagnosis codes into categories.
    """
    try:
        diag = float(diag)
    except:
        return 'Other'

    if 390 <= diag <= 459 or diag == 785:
        return 'Circulatory'
    elif 460 <= diag <= 519 or diag == 786:
        return 'Respiratory'
    elif 520 <= diag <= 579 or diag == 787:
        return 'Digestive'
    elif diag == 250:
        return 'Diabetes'
    elif 800 <= diag <= 999:
        return 'Injury'
    elif 710 <= diag <= 739:
        return 'Musculoskeletal'
    elif 580 <= diag <= 629 or diag == 788:
        return 'Genitourinary'
    elif 140 <= diag <= 239:
        return 'Neoplasms'
    else:
        return 'Other'


def apply_diag_clustering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Application of clustering to diagnosis columns.
    """
    for col in ['diag_1', 'diag_2', 'diag_3']:
        df[col] = df[col].apply(diag_cluster)
    return df
