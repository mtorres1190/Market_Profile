import pandas as pd
import streamlit as st

def parse_household_trends(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    df['Demographic'] = df['Demographic'].str.strip()
    df = df.drop(columns=[col for col in df.columns if '% of total' in col.lower()], errors='ignore')

    # Parse Total Households
    total_households = df[df['Demographic'] == 'Households']
    total_households = (
        total_households
        .melt(id_vars='Demographic', var_name='Year', value_name='Households')
        .drop(columns='Demographic')
    )
    total_households['Households'] = pd.to_numeric(total_households['Households'], errors='coerce')

    # Parse Householder Age
    age_mask = df['Demographic'].between('Householder Age: 15-24', 'Householder Age: 85+')
    age_dist = df[age_mask].copy()
    age_dist['Age Group'] = age_dist['Demographic'].str.replace('Householder Age: ', '', regex=False)
    age_dist = age_dist.drop(columns='Demographic')
    age_dist = age_dist.melt(id_vars='Age Group', var_name='Year', value_name='Households')

    # Parse Household Size
    avg_hh_size = (
        df[df['Demographic'] == 'Average Household Size']
        .melt(id_vars='Demographic', var_name='Year', value_name='AvgSize')
        .drop(columns='Demographic')
    )
    size_mask = df['Demographic'].str.contains('Person Households', na=False)
    size_dist = df[size_mask].copy()
    size_dist['Household Size'] = size_dist['Demographic']
    size_dist = (
        size_dist
        .drop(columns='Demographic')
        .melt(id_vars='Household Size', var_name='Year', value_name='Households')
    )

    # Parse Household Type
    type_mask = df['Demographic'].str.startswith('Household Type:', na=False)
    type_dist = df[type_mask].copy()
    type_dist['Household Type'] = type_dist['Demographic'].str.replace('Household Type: ', '', regex=False)
    type_dist = (
        type_dist
        .drop(columns='Demographic')
        .melt(id_vars='Household Type', var_name='Year', value_name='Households')
    )

    # Parse Income
    avg_income = (
        df[df['Demographic'] == 'Household Income: Average']
        .melt(id_vars='Demographic', var_name='Year', value_name='AvgIncome')
        .drop(columns='Demographic')
    )
    median_income = (
        df[df['Demographic'] == 'Household Income: Median']
        .melt(id_vars='Demographic', var_name='Year', value_name='MedianIncome')
        .drop(columns='Demographic')
    )
    income_mask = (
        df['Demographic'].str.contains('Income', na=False) &
        ~df['Demographic'].str.contains('Average|Median', na=False)
    )
    income_dist = df[income_mask].copy()
    income_dist['Income Bracket'] = income_dist['Demographic']
    income_dist = (
        income_dist
        .drop(columns='Demographic')
        .melt(id_vars='Income Bracket', var_name='Year', value_name='Households')
    )

    # Parse Tenure
    tenure_mask = df['Demographic'].isin([
        'Owner Occupied Housing Units',
        'Renter Occupied Housing Units'
    ])
    tenure_dist = df[tenure_mask].copy()
    tenure_dist['Tenure'] = tenure_dist['Demographic']
    tenure_dist = tenure_dist.drop(columns='Demographic')
    tenure_dist = tenure_dist.melt(id_vars='Tenure', var_name='Year', value_name='Households')

    # Parse Housing Value
    avg_val = (
        df[df['Demographic'] == 'Average Housing Unit Value']
        .melt(id_vars='Demographic', var_name='Year', value_name='AvgValue')
        .drop(columns='Demographic')
    )
    med_val = (
        df[df['Demographic'] == 'Median Housing Unit Value']
        .melt(id_vars='Demographic', var_name='Year', value_name='MedianValue')
        .drop(columns='Demographic')
    )
    value_mask = (
        df['Demographic'].str.contains(r'\$', na=False) &
        ~df['Demographic'].str.contains('Average|Median', na=False)
    )
    value_dist = df[value_mask].copy()
    value_dist['Value Bracket'] = value_dist['Demographic']
    value_dist = (
        value_dist
        .drop(columns='Demographic')
        .melt(id_vars='Value Bracket', var_name='Year', value_name='Households')
    )

    # Parse Year Built
    median_built = (
        df[df['Demographic'] == 'Median Housing Unit Year Built']
        .melt(id_vars='Demographic', var_name='Year', value_name='MedianBuilt')
        .drop(columns='Demographic')
    )
    built_mask = (
        df['Demographic'].str.contains('Built', na=False) &
        ~df['Demographic'].str.contains('Median', na=False)
    )
    built_dist = df[built_mask].copy()
    built_dist['Year Built Range'] = built_dist['Demographic']
    built_dist = (
        built_dist
        .drop(columns='Demographic')
        .melt(id_vars='Year Built Range', var_name='Year', value_name='Households')
    )

    return {
        'total_households': total_households,
        'householder_age': age_dist,
        'average_household_size': avg_hh_size,
        'household_size_distribution': size_dist,
        'household_type_distribution': type_dist,
        'income_metrics': {'average': avg_income, 'median': median_income},
        'income_distribution': income_dist,
        'tenure_distribution': tenure_dist,
        'housing_value_metrics': {'average': avg_val, 'median': med_val},
        'housing_value_distribution': value_dist,
        'year_built_metrics': median_built,
        'year_built_distribution': built_dist
    }

