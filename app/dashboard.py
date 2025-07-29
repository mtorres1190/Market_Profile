import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os

# 🛠 Developer Mode (set to False to enable auto-loading of test files and disable firewall)
AUTO_UPLOAD = False  # Enable to auto-load dev files
ENABLE_FIREWALL = False  # Enable to require access code


# Set your 4-digit access code
ACCESS_CODE = "2025"

# Apply firewall gate if enabled
if ENABLE_FIREWALL:
    if "authorized" not in st.session_state:
        st.session_state.authorized = False

    if not st.session_state.authorized:
        st.markdown("### 🔒 Enter Access Code to Continue")
        code_input = st.text_input("Enter Access Code", type="password", max_chars=4)

        if st.button("Submit"):
            if code_input == ACCESS_CODE:
                st.session_state.authorized = True
                st.rerun()
            else:
                st.error("❌ Invalid code. Please try again.")

        st.stop()  # Halt app until access is granted

# Add project root to path to locate scripts folder one level up
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.preprocess import parse_household_trends
from segment_definitions import segment_descriptions
from utils.mortgage_config import MORTGAGE_RATE

st.set_page_config(page_title="Dashboard", layout="wide")
st.markdown("<h1 style='margin-bottom: 0.5rem;'>Market Profile Dashboard</h1>", unsafe_allow_html=True)

REQUIRED_HT_COLUMNS = {'Demographic', '2020', '% OF TOTAL', '2024', '% OF TOTAL', '2029', '% OF TOTAL'}
REQUIRED_CS_COLUMNS = {'RANK', 'Name', '# 0F HOUSEHOLDS', '% OF TOTAL'}

def validate_household_file(file):
    try:
        file.seek(0)
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        return set(REQUIRED_HT_COLUMNS).issubset(set(df.columns))
    except Exception:
        return False

def validate_consumer_file(file):
    try:
        file.seek(0)
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        return set(REQUIRED_CS_COLUMNS).issubset(set(df.columns))
    except Exception:
        return False

for key in ["ht_file", "household_data", "cs_file", "consumer_data"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Auto-load files in developer mode
if AUTO_UPLOAD:
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ht_path = os.path.join(BASE_DIR, "..", "data", "uploads", "HouseholdTrends - Developer.csv")
    cs_path = os.path.join(BASE_DIR, "..", "data", "uploads", "ConsumerSegmentation - Developer.csv")
    import io
    with open(ht_path, "r", encoding="utf-8") as ht_f:
        st.session_state.ht_file = ht_f.read()
        ht_buffer = io.StringIO(st.session_state.ht_file)
        st.session_state.household_data = parse_household_trends(ht_buffer)
    import io
    with open(cs_path, "r", encoding="utf-8") as cs_f:
        st.session_state.cs_file = cs_f.read()
        cs_buffer = io.StringIO(st.session_state.cs_file)
        st.session_state.consumer_data = pd.read_csv(cs_buffer)
else:
    col1, col2 = st.columns(2)
    with col1:
        ht_input = st.file_uploader("Upload Household Trends CSV", type="csv", key="ht_upload")
    with col2:
        cs_input = st.file_uploader("Upload Consumer Segmentation CSV", type="csv", key="cs_upload")

    if ht_input is None:
        st.session_state.ht_file = None
        st.session_state.household_data = None
    if cs_input is None:
        st.session_state.cs_file = None
        st.session_state.consumer_data = None

    if ht_input:
        if validate_household_file(ht_input):
            ht_input.seek(0)
            st.session_state.ht_file = ht_input
            st.session_state.household_data = parse_household_trends(ht_input)
        else:
            st.error("❌ Invalid Household Trends file format.")

    if cs_input:
        if validate_consumer_file(cs_input):
            cs_input.seek(0)
            st.session_state.cs_file = cs_input
            st.session_state.consumer_data = pd.read_csv(cs_input)
        else:
            st.error("❌ Invalid Consumer Segmentation file format.")


valid = (
    st.session_state.get("ht_file") is not None
    and st.session_state.get("cs_file") is not None
    and st.session_state.get("household_data") is not None
    and st.session_state.get("consumer_data") is not None
)

if not valid:
    st.markdown("<div style='margin-bottom: 1rem;'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='background-color: #fff3cd; padding: 0.75rem 1rem; border-radius: 0.5rem; color: #856404; font-size: 0.9rem;'>\n        ⚠️ Please upload valid versions of <strong>both required files</strong> to continue.\n        </div>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Households Growth Chart
household_data = st.session_state.get("household_data")
consumer_data = st.session_state.get("consumer_data")

if valid and household_data and 'total_households' in household_data:
    total_df = household_data['total_households'].copy()
    total_df['Year'] = total_df['Year'].astype(int)
    total_df = total_df.sort_values('Year')
    total_df['Households'] = pd.to_numeric(total_df['Households'], errors='coerce')

    if len(total_df) >= 3:
        values = total_df['Households'].values
        pct_2024 = ((values[1] - values[0]) / values[0]) * 100
        pct_2029 = ((values[2] - values[1]) / values[1]) * 100

        def pct_label(pct):
            sign = "+" if pct >= 0 else "-"
            return f"{sign}{abs(pct):.1f}%"

        fig = go.Figure()
        custom_texts = []
        for i in range(len(total_df)):
            year = total_df.iloc[i]['Year']
            hh = int(total_df.iloc[i]['Households'])
            if i == 0:
                change = "N/A"
            else:
                prev = total_df.iloc[i-1]['Households']
                delta = ((hh - prev) / prev) * 100
                sign = "+" if delta >= 0 else "-"
                change = f"{sign}{abs(delta):.1f}%"
            custom_texts.append(f"Year: {year}<br>Households: {hh:,}<br>% Change: {change}")

        fig.add_trace(go.Scatter(
            x=total_df['Year'],
            y=total_df['Households'],
            fill='tozeroy',
            mode='lines+markers',
            line=dict(color='royalblue'),
            name='Total Households',
            hoverinfo='text',
            text=custom_texts
        ))
        for i, year in enumerate(total_df['Year']):
            fig.add_shape(
                type="line",
                x0=year,
                y0=0,
                x1=year,
                y1=values[i],
                line=dict(color="gray", width=1, dash="dash")
            )
        fig.add_annotation(x=2024, y=values[1],
            text=pct_label(pct_2024), showarrow=True, arrowhead=1, ax=0, ay=-40,
            font=dict(size=18))
        fig.add_annotation(x=2029, y=values[2],
            text=pct_label(pct_2029), showarrow=True, arrowhead=1, ax=0, ay=-40,
            font=dict(size=18))
        fig.update_layout(
            title="Households Growth",
            xaxis_title="Year",
            yaxis_title="Households",
            height=400,
            xaxis=dict(tickmode='array', tickvals=list(total_df['Year']))
        )

# Households: Own vs. Rent Chart
        viz1, viz2 = st.columns(2)
        with viz1:
            st.plotly_chart(fig, use_container_width=True)

        with viz2:
            tenure_df = household_data['tenure_distribution'].copy()
            tenure_df = tenure_df[tenure_df['Year'].isin(['2020', '2024'])]
            tenure_df['Households'] = pd.to_numeric(tenure_df['Households'], errors='coerce')

            tenure_df['Tenure'] = tenure_df['Tenure'].replace({
                'Owner Occupied Housing Units': 'Own',
                'Renter Occupied Housing Units': 'Rent'
            })

            outer = tenure_df[tenure_df['Year'] == '2024']
            inner = tenure_df[tenure_df['Year'] == '2020']

            fig_nested = go.Figure()

            fig_nested.add_trace(go.Pie(
                labels=[f"Own (2020)" if label == 'Own' else "Rent (2020)" for label in inner['Tenure']],
                values=inner['Households'],
                name='2020',
                hole=0.6,
                direction='clockwise',
                sort=False,
                texttemplate='%{percent:.1%}',
                textposition='inside',
                insidetextorientation='radial',
                hoverinfo='label+value+percent',
                textfont=dict(size=18, color='white'),
                marker=dict(colors=["#51b0f5", "#aebdd4"], line=dict(color='#FFFFFF', width=2)),
                opacity=1.0,
                domain={'x': [0.15, 0.85], 'y': [0.15, 0.85]},
                showlegend=True
            ))

            fig_nested.add_trace(go.Pie(
                labels=[f"Own (2024)" if label == 'Own' else "Rent (2024" for label in outer['Tenure']],
                values=outer['Households'],
                name='2024',
                hole=0.7,
                direction='clockwise',
                sort=False,
                texttemplate='%{percent}',
                textposition='inside',
                insidetextorientation='radial',
                hoverinfo='label+value+percent',
                textfont=dict(size=16, color='white'),
                marker=dict(colors=['#1f77b4', "#8698b1"], line=dict(color='#FFFFFF', width=2)),
                domain={'x': [0, 1], 'y': [0, 1]},
                showlegend=True
            ))

            fig_nested.update_layout(
                title_text="Households: Own vs. Rent",
                legend_traceorder='normal',
                height=400
            )

            st.plotly_chart(fig_nested, use_container_width=True)

# Add Tree Map Chart
if valid and consumer_data is not None and household_data is not None:
    bubble_df = consumer_data.copy()
    bubble_df.columns = bubble_df.columns.str.strip()

    if 'Name' in bubble_df.columns and '# 0F HOUSEHOLDS' in bubble_df.columns:
        bubble_df = bubble_df[bubble_df['Name'].str.lower() != 'total']
        bubble_df = bubble_df.drop_duplicates(subset=['Name'])
        bubble_df['Size'] = pd.to_numeric(bubble_df['# 0F HOUSEHOLDS'], errors='coerce')
        bubble_df = bubble_df.dropna(subset=['Size', 'Name'])

        group_mapping = {
            'Entry Level Suburban': 'Entry Level',
            'Entry Level Urban': 'Entry Level',
            'Renters': 'Renters',
            'Renters Near Term Buyers': 'Renters',
            'Family Life Young Children': 'Family Life',
            'Family Life School Age Children': 'Family Life',
            'Active Adult Elite': 'Active Adult',
            'Active Adult Entry Level': 'Active Adult',
            'Active Adult Feature and Location': 'Active Adult',
            'Simple Life Affluent No Children': 'Simple Life',
            'Simple Life Moderate Income With Children': 'Simple Life',
            'Simple Life Moderate Income No Children': 'Simple Life',
            'Feature and Location': 'Feature and Location',
            'Elite': 'Elite'
        }

        group_colors = {
            'Entry Level': '#1f77b4',
            'Renters': '#ff7f0e',
            'Family Life': '#2ca02c',
            'Active Adult': '#d62728',
            'Simple Life': '#9467bd',
            'Elite': '#8c564b',
            'Feature and Location': '#e377c2',
            'Unclassified': '#7f7f7f'
        }

        bubble_df['Group'] = bubble_df['Name'].map(group_mapping).fillna('Unclassified')
        bubble_df['Color'] = bubble_df['Group'].map(group_colors).fillna('#7f7f7f')

        if bubble_df.empty:
            st.warning("No valid segments available to display in the treemap.")
            st.stop()

        group_nodes = pd.DataFrame({
            'ids': ['group_' + grp for grp in bubble_df['Group'].unique()],
            'labels': [
                f"{grp} ({bubble_df[bubble_df['Group'] == grp]['Size'].sum() / bubble_df['Size'].sum():.1%})"
                for grp in bubble_df['Group'].unique()
            ],
            'parents': ['All Segments'] * bubble_df['Group'].nunique(),
            'values': [bubble_df[bubble_df['Group'] == grp]['Size'].sum() for grp in bubble_df['Group'].unique()],
            'colors': ['#cccccc'] * bubble_df['Group'].nunique()
        })

        leaf_nodes = bubble_df[['Name', 'Group', 'Size', 'Color']].rename(columns={
            'Name': 'ids',
            'Size': 'values',
            'Color': 'colors'
        })
        leaf_nodes['labels'] = leaf_nodes['ids']
        leaf_nodes['parents'] = 'group_' + bubble_df['Group']

        all_nodes = pd.concat([
            pd.DataFrame({
                'ids': ['All Segments'],
                'labels': ['All Segments'],
                'parents': [''],
                'values': [bubble_df['Size'].sum()],
                'colors': ['#eeeeee']
            }),
            group_nodes,
            leaf_nodes
        ])

        description_map = {
            row['ids']: segment_descriptions.get(row['ids'].strip(), 'No description available') or 'No description available'
            for _, row in leaf_nodes.iterrows()
        }

        fig_treemap = go.Figure(go.Treemap(
            ids=all_nodes['ids'],
            labels=all_nodes['labels'],
            parents=all_nodes['parents'],
            values=all_nodes['values'],
            marker=dict(colors=all_nodes['colors']),
            textinfo="label+percent parent+percent root",
            hovertext=[
            f"{label}<br><br>{description_map[id].replace(chr(10), '<br>')}"
            if id in description_map else label
            for label, id in zip(all_nodes['labels'], all_nodes['ids'])
        ],
            hoverinfo="text",
            branchvalues="total"
        ))

        fig_treemap.update_layout(
            title="Segment Distribution by Group",
            height=550,
            margin=dict(t=40, l=10, r=10, b=10)
        )

        st.plotly_chart(fig_treemap, use_container_width=True)
    else:
        st.error("❌ Required columns missing from Consumer Segmentation file.")

# Predefine variables to avoid unbound errors
lennar_price = None
required_income_lennar = None

# Helper function to calculate required income

def calculate_required_income(price, mortgage_rate):
    down_payment = 0.03
    loan_amount = price * (1 - down_payment)
    monthly_rate = mortgage_rate / 12
    term_months = 30 * 12

    # Monthly mortgage (P&I)
    monthly_mortgage = loan_amount * (monthly_rate * (1 + monthly_rate) ** term_months) / \
                       ((1 + monthly_rate) ** term_months - 1)

    # Monthly property taxes and insurance
    monthly_taxes = price * 0.0173 / 12
    monthly_insurance = price * 0.004 / 12

    # Monthly FHA mortgage insurance premium (PMI)
    monthly_pmi = loan_amount * 0.0055 / 12

    # Total monthly housing cost (PITI + PMI)
    total_housing_payment = monthly_mortgage + monthly_taxes + monthly_insurance + monthly_pmi

    # Use 31% front-end DTI per FHA guidelines
    required_income_monthly = total_housing_payment / 0.31
    return required_income_monthly * 12, total_housing_payment


# Add Lennar Price Input
if valid:
    st.markdown("""
        <style>
        div[data-testid=\"stTextInput\"] input {
            border: 1px solid #ccc !important;
            box-shadow: none !important;
            outline: none !important;
        }
        div[data-testid=\"stTextInput\"] input:focus {
            border: 1px solid #ccc !important;
            box-shadow: none !important;
            outline: none !important;
        }
        div[data-testid=\"stTextInput\"] input:focus:invalid {
            border: 1px solid #ccc !important;
            box-shadow: none !important;
        }
        </style>
    """, unsafe_allow_html=True)

    if "price_input" not in st.session_state:
        st.session_state.price_input = ""

    def format_price():
        raw = ''.join(ch for ch in st.session_state.price_input if ch.isdigit())
        if raw:
            formatted = f"${int(raw):,}"
            st.session_state.price_input = formatted

    st.markdown("""
        <div style='font-weight: 700; font-size: 1.15rem; margin-bottom: 0.0rem;'>
            Enter the Average Lennar home price in this area
        </div>
    """, unsafe_allow_html=True)

    st.text_input(
        label="",
        placeholder="e.g. $325,000",
        key="price_input",
        on_change=format_price
    )

    try:
        clean_numeric = ''.join(ch for ch in st.session_state.price_input if ch.isdigit())
        if clean_numeric:
            lennar_price = float(clean_numeric)
            required_income_lennar, _ = calculate_required_income(lennar_price, MORTGAGE_RATE)
    except Exception:
        lennar_price = None


# Additional Market Insights row
household_data = st.session_state.get("household_data")

if valid and household_data is not None and isinstance(household_data, dict):
    st.subheader("Additional Market Insights")
    col5, col6 = st.columns(2)

    # Household Income Distribution Chart
    with col5:
        if 'income_distribution' in household_data:
            income_df_all = household_data['income_distribution'].copy()
            income_df_all.columns = income_df_all.columns.str.strip()

            column_name = None
            if 'Demographic' in income_df_all.columns:
                column_name = 'Demographic'
            elif 'Income Bracket' in income_df_all.columns:
                column_name = 'Income Bracket'

            if column_name is None:
                st.error("❌ Required income bracket column not found in income_distribution data.")
                st.write("Columns found:", income_df_all.columns.tolist())
            else:
                income_brackets = {
                    'Household Income: Less than $25k': 25000,
                    'Household Income: $25k - $50k': 50000,
                    'Household Income: $50k - $75k': 75000,
                    'Household Income: $75k - $100k': 100000,
                    'Household Income: $100k - $125k': 125000,
                    'Household Income: $125k - $150k': 150000,
                    'Household Income: $150k - $200k': 200000,
                    'Household Income: Above $200k': 225000
                }

                def prepare_income_df(df, year):
                    df_year = df[(df['Year'] == year) & (df[column_name].isin(income_brackets.keys()))].copy()
                    df_year.loc[df_year[column_name] == 'Household Income: $150k - $200k', 'Households'] = pd.to_numeric(df_year.loc[df_year[column_name] == 'Household Income: $150k - $200k', 'Households'], errors='coerce') / 2
                    df_year['Households'] = pd.to_numeric(df_year['Households'], errors='coerce')
                    df_year['x'] = df_year[column_name].map(income_brackets)
                    df_year = df_year.sort_values('x')
                    zero_point = pd.DataFrame({'x': [0], 'Households': [0]})
                    return pd.concat([zero_point, df_year], ignore_index=True)

                income_df_2024 = prepare_income_df(income_df_all, '2024')
                income_df_2020 = prepare_income_df(income_df_all, '2020')

                avg_row = household_data['income_metrics']['average']
                med_row = household_data['income_metrics']['median']

                avg_income = pd.to_numeric(avg_row[avg_row['Year'] == '2024']['AvgIncome'].values[0], errors='coerce') if not avg_row.empty else None
                med_income = pd.to_numeric(med_row[med_row['Year'] == '2024']['MedianIncome'].values[0], errors='coerce') if not med_row.empty else None

                max_y = max(income_df_2024['Households'].max(), income_df_2020['Households'].max()) * 1.1
                max_x = 225000
                income_ticks = list(range(0, max_x + 1, 50000)) + [225000]

                fig_income = go.Figure()

                fig_income.add_trace(go.Scatter(
                    x=income_df_2024['x'],
                    y=income_df_2024['Households'],
                    fill='tozeroy',
                    mode='lines+markers',
                    line=dict(color='seagreen', shape='spline', smoothing=1.3),
                    name='2024',
                    hovertemplate='2024 Income: %{x:$,.0f}<br>Households: %{y:,}<extra></extra>'
                ))

                fig_income.add_trace(go.Scatter(
                    x=income_df_2020['x'],
                    y=income_df_2020['Households'],
                    mode='lines+markers',
                    line=dict(color='purple', dash='dash', shape='spline', smoothing=1.3),
                    name='2020',
                    hovertemplate='2020 Income: %{x:$,.0f}<br>Households: %{y:,}<extra></extra>'
                ))

                if avg_income is not None:
                    fig_income.add_trace(go.Scatter(
                        x=[avg_income, avg_income],
                        y=[0, max_y],
                        mode='lines',
                        name=f'Avg: ${int(avg_income):,}',
                        hovertemplate=f'Avg: ${int(avg_income):,}<extra></extra>',
                        line=dict(color='blue', dash='dot'),
                        showlegend=True
                    ))

                if med_income is not None:
                    fig_income.add_trace(go.Scatter(
                        x=[med_income, med_income],
                        y=[0, max_y],
                        mode='lines',
                        name=f'Med: ${int(med_income):,}',
                        hovertemplate=f'Med: ${int(med_income):,}<extra></extra>',
                        line=dict(color='orange', dash='dot'),
                        showlegend=True
                    ))

                if required_income_lennar is not None:
                    fig_income.add_trace(go.Scatter(
                        x=[required_income_lennar, required_income_lennar],
                        y=[0, max_y],
                        mode='lines',
                        name=f'Required: ${int(required_income_lennar):,}',
                        hovertemplate=f'Required: ${int(required_income_lennar):,}<extra></extra>',
                        line=dict(color='gray', dash='dash'),
                        showlegend=True
                    ))

                fig_income.update_layout(
                    title="Household Income Distribution",
                    xaxis_title="Household Income",
                    yaxis_title="Households",
                    xaxis=dict(
                        tickmode='array',
                        tickvals=income_ticks,
                        ticktext=[">$200k" if x == 225000 else f"${x/1000:.0f}k" for x in income_ticks],
                        ticks='outside'
                    ),
                    yaxis=dict(tickformat=','),
                    height=450
                )

                st.plotly_chart(fig_income, use_container_width=True)


    # Household Value Distribution Chart
    with col6:
        if 'housing_value_distribution' in household_data:
            value_df_all = household_data['housing_value_distribution'].copy()
            value_df_all.columns = value_df_all.columns.str.strip()

            value_brackets = {
                'Housing Unit Value Less Than $100k': 100000,
                'Housing Unit Value $100k-$200k': 200000,
                'Housing Unit Value $200k-$250k': 250000,
                'Housing Unit Value $250k-$300k': 300000,
                'Housing Unit Value $300k-$400k': 400000,
                'Housing Unit Value $400k-$500k': 500000,
                'Housing Unit Value $500k-$750k': 750000,
                'Housing Unit Value $750k-$1m': 800000,
                'Housing Unit Value $1m-$1.5m': 800000,
                'Housing Unit Value $1.5m-$2m': 800000,
                'Housing Unit Value Above $2m': 800000,
                'Housing Unit Value > $750k': 800000
            }

            def prepare_value_df(df, year):
                df = df.copy()
                df['Value Bracket'] = df['Value Bracket'].replace({
                    'Housing Unit Value $750k-$1m': 'Housing Unit Value > $750k',
                    'Housing Unit Value $1m-$1.5m': 'Housing Unit Value > $750k',
                    'Housing Unit Value $1.5m-$2m': 'Housing Unit Value > $750k',
                    'Housing Unit Value Above $2m': 'Housing Unit Value > $750k'
                })
                df_year = df[(df['Year'] == year) & (df['Value Bracket'].isin(value_brackets.keys()) | (df['Value Bracket'] == 'Housing Unit Value > $750k'))].copy()
                df_year['Households'] = pd.to_numeric(df_year['Households'], errors='coerce')
                df_year = df_year.groupby('Value Bracket', as_index=False).agg({'Households': 'sum'})

                adjustments = {
                    'Housing Unit Value Less Than $100k': 2,
                    'Housing Unit Value $100k-$200k': 2,
                    'Housing Unit Value $300k-$400k': 2,
                    'Housing Unit Value $400k-$500k': 2,
                    'Housing Unit Value $500k-$750k': 5
                }

                df_year['Households'] = df_year['Households'].astype(float)

                for bracket, divisor in adjustments.items():
                    df_year.loc[df_year['Value Bracket'] == bracket, 'Households'] = (
                        pd.to_numeric(df_year.loc[df_year['Value Bracket'] == bracket, 'Households'], errors='coerce') / divisor
                    ).astype(float)

                df_year['x'] = df_year['Value Bracket'].map(lambda vb: value_brackets.get(vb, 800000))
                df_year['Hover Label'] = df_year['Value Bracket'].apply(
                    lambda vb: '> $750k' if '>$750k' in vb or vb.startswith('Housing Unit Value > $750k') else f"${value_brackets.get(vb, 800000):,.0f}"
                )
                df_year = df_year.sort_values('x')
                zero_point = pd.DataFrame({'x': [0], 'Households': [0], 'Hover Label': ['$0']})
                return pd.concat([zero_point, df_year], ignore_index=True)

            value_df_2024 = prepare_value_df(value_df_all, '2024')
            value_df_2020 = prepare_value_df(value_df_all, '2020')

            value_metrics = household_data.get('housing_value_metrics', {})
            avg_value = pd.to_numeric(value_metrics.get('average', pd.DataFrame()).query("Year == '2024'")['AvgValue'].values[0], errors='coerce') if 'average' in value_metrics else None
            med_value = pd.to_numeric(value_metrics.get('median', pd.DataFrame()).query("Year == '2024'")['MedianValue'].values[0], errors='coerce') if 'median' in value_metrics else None

            value_df_2024['Households'] = pd.to_numeric(value_df_2024['Households'], errors='coerce')
            value_df_2020['Households'] = pd.to_numeric(value_df_2020['Households'], errors='coerce')
            max_y_val = max(value_df_2024['Households'].max(skipna=True), value_df_2020['Households'].max(skipna=True)) * 1.1
            max_val_x = 800000
            value_ticks = list(range(0, max_val_x, 100000)) + [800000]

            fig_value = go.Figure()
            fig_value.add_trace(go.Scatter(
                x=value_df_2024['x'],
                y=value_df_2024['Households'],
                fill='tozeroy',
                mode='lines+markers',
                line=dict(color='seagreen', shape='spline', smoothing=1.3),
                name='2024',
                customdata=value_df_2024['Hover Label'],
                hovertemplate='2024 Value: %{customdata}<br>Households: %{y:,}<extra></extra>'
            ))
            fig_value.add_trace(go.Scatter(
                x=value_df_2020['x'],
                y=value_df_2020['Households'],
                mode='lines+markers',
                line=dict(color='purple', dash='dash', shape='spline', smoothing=1.3),
                name='2020',
                customdata=value_df_2020['Hover Label'],
                hovertemplate='2020 Value: %{customdata}<br>Households: %{y:,}<extra></extra>'
            ))

            if avg_value is not None:
                fig_value.add_trace(go.Scatter(
                    x=[avg_value, avg_value],
                    y=[0, max_y_val],
                    mode='lines',
                    name=f'Avg: ${int(avg_value):,}',
                    hovertemplate=f'Avg: ${int(avg_value):,}<extra></extra>',
                    line=dict(color='blue', dash='dot'),
                    showlegend=True
                ))

            if med_value is not None:
                fig_value.add_trace(go.Scatter(
                    x=[med_value, med_value],
                    y=[0, max_y_val],
                    mode='lines',
                    name=f'Med: ${int(med_value):,}',
                    hovertemplate=f'Med: ${int(med_value):,}<extra></extra>',
                    line=dict(color='orange', dash='dot'),
                    showlegend=True
                ))

            if lennar_price is not None:
                fig_value.add_trace(go.Scatter(
                    x=[lennar_price, lennar_price],
                    y=[0, max_y_val],
                    mode='lines',
                    name=f'Home Price: ${int(lennar_price):,}',
                    hovertemplate=f'Home Price: ${int(lennar_price):,}<extra></extra>',
                    line=dict(color='gray', dash='dash'),
                    showlegend=True
                ))

            fig_value.update_layout(
                title="Household Value Distribution",
                xaxis_title="Household Value",
                yaxis_title="Households",
                xaxis=dict(
                    tickmode='array',
                    tickvals=value_ticks,
                    ticktext=[f"${x/1000:.0f}k" if x < 800000 else "> $750k" for x in value_ticks],
                    tickformat=',',
                    ticks='outside'
                ),
                yaxis=dict(tickformat=','),
                height=450
            )

            st.plotly_chart(fig_value, use_container_width=True)


    # Home Affordability Index Chart
if valid and household_data is not None and isinstance(household_data, dict):
    avg_income_row = household_data['income_metrics']['median']
    avg_value_row = household_data['housing_value_metrics']['median']

    if not avg_income_row.empty and not avg_value_row.empty:
        income_2024 = pd.to_numeric(avg_income_row[avg_income_row['Year'] == '2024']['MedianIncome'].values[0], errors='coerce')
        value_2024 = pd.to_numeric(avg_value_row[avg_value_row['Year'] == '2024']['MedianValue'].values[0], errors='coerce')

        if pd.notnull(income_2024) and pd.notnull(value_2024):
            required_income, _ = calculate_required_income(value_2024, MORTGAGE_RATE)
            hai = (income_2024 / required_income) * 100

            hai_fig = go.Figure()

            gradient_colors = [
                [0.0, "#cc0000"],
                [0.25, "#ff9999"],
                [0.5, "#f2f2f2"],
                [0.75, "#a6e6a6"],
                [1.0, "#008000"]
            ]

            x_vals = list(range(201))
            z_vals = [[x / 200 for x in x_vals], [x / 200 for x in x_vals]]

            hai_fig.add_trace(go.Heatmap(
                z=z_vals,
                x=x_vals,
                y=[0, 1],
                colorscale=gradient_colors,
                showscale=False,
                hoverinfo='skip'
            ))

            hai_fig.add_shape(
                type="line",
                x0=100,
                x1=100,
                y0=0,
                y1=1,
                line=dict(color="gray", dash="dash", width=2)
            )

            hai_fig.add_shape(
                type="line",
                x0=hai,
                x1=hai,
                y0=0.05,
                y1=0.95,
                line=dict(color="black", width=3)
            )

            hai_fig.add_annotation(
                x=hai,
                y=1.08,
                text=f"HAI: {hai:.1f}",
                showarrow=False,
                font=dict(size=16, color="black"),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=4
            )

            interpretation_addon = ""
            if required_income_lennar is not None:
                hai_lennar = (income_2024 / required_income_lennar) * 100

                hai_fig.add_shape(
                    type="line",
                    x0=hai_lennar,
                    x1=hai_lennar,
                    y0=0.05,
                    y1=0.95,
                    line=dict(color="black", dash="dot", width=2)
                )

                hai_fig.add_annotation(
                    x=hai_lennar,
                    y=1.08,
                    text=f"Lennar HAI: {hai_lennar:.1f}",
                    showarrow=False,
                    font=dict(size=15, color="black"),
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=4
                )

                interpretation_addon = f"<br>Given Lennar's average price of <strong>${lennar_price:,.0f}</strong> in this market, the HAI of Lennar homes is <strong>{hai_lennar:.1f}</strong>."

            # Chart title with polished hover tooltip
            st.markdown("""
                <style>
                .tooltip-container {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    gap: 0.4rem;
                    font-weight: 600;
                    font-size: 1.25rem;
                    margin-bottom: 0.25rem;
                }
                .tooltip {
                    position: relative;
                    display: inline-block;
                    cursor: pointer;
                }
                .tooltip .tooltiptext {
                    visibility: hidden;
                    width: 280px;
                    background-color: #ffffff;
                    color: #333;
                    text-align: left;
                    border-radius: 8px;
                    padding: 1rem;
                    position: absolute;
                    z-index: 10;
                    top: 140%;
                    left: 50%;
                    transform: translateX(-50%);
                    box-shadow: 0px 4px 16px rgba(0, 0, 0, 0.15);
                    font-size: 0.9rem;
                    line-height: 1.4;
                }
                .tooltip:hover .tooltiptext {
                    visibility: visible;
                }
                .tooltiptext ul {
                    margin: 0.25rem 0 0 1rem;
                    padding: 0;
                }
                .tooltiptext li {
                    margin-bottom: 0.5rem;
                }
                </style>
                <div class="tooltip-container">
                    Home Affordability Index (HAI)
                    <div class="tooltip">💡
                        <div class="tooltiptext">
                            <strong>HAI Assumptions:</strong>
                            <ul>
                                <li>3% Down Payment</li>
                                <li>30-Year Fixed Mortgage (FHA)</li>
                                <li>3.99% Interest Rate</li>
                                <li>1.73% Property Tax</li>
                                <li>0.4% Homeowners Insurance</li>
                                <li>0.55% Private Mortgage Insurance (PMI)</li>
                                <li>31% Front-End DTI Limit</li>
                            </ul>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)


            hai_fig.update_layout(
                xaxis=dict(
                    range=[0, 200],
                    tickvals=[0, 50, 100, 150, 200],
                    ticktext=["0", "50", "100 (Balanced)", "150", "200"],
                    title=dict(text="HAI Value", font=dict(size=14)),
                    tickfont=dict(size=12),
                    showgrid=False,
                    showline=True,
                    linewidth=1,
                    linecolor='black',
                    fixedrange=True
                ),
                yaxis=dict(
                    visible=False,
                    fixedrange=True
                ),
                height=180,
                margin=dict(t=40, b=30, l=30, r=30),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )

            st.plotly_chart(hai_fig, use_container_width=True)

            st.markdown(f"""
            <div style='margin-top: -0.5rem; font-size: 1.05rem; color: #333;'>
                <strong>Interpretation:</strong> The income required to afford the median home in this market is <strong>${required_income:,.0f}</strong>. However, the median income of the market is actually <strong>${income_2024:,.0f}</strong>. This is <strong>{hai:.1f}%</strong> of the income needed, which gives us a HAI value of <strong>{hai:.1f}</strong>.{interpretation_addon}
            </div>
            """, unsafe_allow_html=True)





# To run the Streamlit App
# streamlit run "C:\\Users\\MiTorres\\OneDrive - Lennar Azure AD\\Files\\Other Projects\\Demographics\\app\\dashboard.py"
# https://market-profile.streamlit.app/
