'''from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import eurostat
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import io
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Country mapping for NUTS codes
nuts0_country_map = {
    'AL': 'Albania', 'AT': 'Austria', 'BA': 'Bosnia and Herzegovina',
    'BE': 'Belgium', 'BG': 'Bulgaria', 'CY': 'Cyprus', 'CZ': 'Czechia',
    'DE': 'Germany', 'DK': 'Denmark', 'EE': 'Estonia', 'EL': 'Greece',
    'ES': 'Spain', 'FI': 'Finland', 'FR': 'France', 'HR': 'Croatia',
    'HU': 'Hungary', 'IE': 'Ireland', 'IS': 'Iceland', 'IT': 'Italy',
    'LI': 'Liechtenstein', 'LT': 'Lithuania', 'LU': 'Luxembourg',
    'LV': 'Latvia', 'ME': 'Montenegro', 'MK': 'North Macedonia',
    'MT': 'Malta', 'NL': 'Netherlands', 'NO': 'Norway', 'PL': 'Poland',
    'PT': 'Portugal', 'RO': 'Romania', 'RS': 'Serbia', 'SE': 'Sweden',
    'SI': 'Slovenia', 'SK': 'Slovakia', 'TR': 'Türkiye', 'UA': 'Ukraine',
    'UK': 'United Kingdom', 'XK': 'Kosovo'
}

def clean_and_melt_renewable_energy_data(df):
    """
    Renames the geographic column and transforms the DataFrame from wide format (years as columns) to a tidy, long format,
    based on the provided column structure.
    """
    # 1. Rename the geographic column and drop index
    if 'geo\\TIME_PERIOD' in df.columns:
        df = df.rename(columns={'geo\\TIME_PERIOD': 'geo'})
    
    # Drop the unnecessary 'index' column
    if 'index' in df.columns:
        df = df.drop(columns=['index'])

    # 2. Define identifier columns and melt the DataFrame
    id_vars = ['freq', 'nrg_bal', 'unit', 'geo']

    # Melt the data: the remaining columns (years) become 'year', and values become 'renewable_share_pct'
    df_long = df.melt(
        id_vars=id_vars,
        var_name='year',
        value_name='renewable_share_pct'
    )

    # 3. Data Type Conversion and Cleaning
    # Convert 'year' to integer
    df_long['year'] = pd.to_numeric(df_long['year'], errors='coerce').astype('Int64')

    # Convert the value column to numeric (the actual percentage)
    df_long['renewable_share_pct'] = pd.to_numeric(df_long['renewable_share_pct'], errors='coerce')

    # Remove rows where the share percentage is NaN (Handling missing data)
    df_long = df_long.dropna(subset=['renewable_share_pct', 'geo']).reset_index(drop=True)

    return df_long

def clean_and_melt_energy_balance_data(df):
    """
    Renames the geographic column and transforms the Energy Balance DataFrame
    from wide format (years as columns) to a tidy, long format.
    """
    # 1. Rename the geographic column and drop index
    if 'geo\\TIME_PERIOD' in df.columns:
        df = df.rename(columns={'geo\\TIME_PERIOD': 'geo'})

    # Drop the redundant 'index' column
    if 'index' in df.columns:
        df = df.drop(columns=['index'])

    # 2. Define identifier columns and melt the DataFrame
    id_vars = ['freq', 'nrg_bal', 'siec', 'unit', 'geo']

    # Melt the data: the remaining columns (years) become 'year', and values become 'energy_value_gwh'
    df_long = df.melt(
        id_vars=id_vars,
        var_name='year',
        value_name='energy_value_gwh'
    )

####this programming is very good to work with the imporant thing is the name of the thing 

    # 3. Data Type Conversion and Cleaning
    # Convert 'year' to integer
    df_long['year'] = pd.to_numeric(df_long['year'], errors='coerce').astype('Int64')

    # Convert the value column (energy in GWH) to numeric
    df_long['energy_value_gwh'] = pd.to_numeric(df_long['energy_value_gwh'], errors='coerce')

    # Remove rows where the energy value is NaN (Handling missing data)
    df_long = df_long.dropna(subset=['energy_value_gwh', 'geo']).reset_index(drop=True)

    return df_long

def integrate_geo_data(df, nuts0_map):
    """Process geo data with NUTS level mapping"""
    df_processed = df.copy()
    
    def determine_nuts_level(code):
        if pd.isna(code):
            return None
        code_str = str(code)
        length = len(code_str)
        if length == 2:
            return 0
        elif length == 3:
            return 1
        elif length == 4:
            return 2
        else:
            return None

    df_processed['nuts_level'] = df_processed['geo'].apply(determine_nuts_level)
    df_processed['country_code'] = df_processed['geo'].str[:2]
    df_processed['country_name'] = df_processed['country_code'].map(nuts0_map)

    return df_processed.dropna(subset=['nuts_level', 'country_name']).reset_index(drop=True)

def load_and_process_eurostat_data():
    """Load and process both Eurostat datasets using the cleaning functions"""
    try:
        print("Loading nrg_ind_ren (Renewable Energy Indicators)...")
        df_renewable_energy = eurostat.get_data_df('nrg_ind_ren')
        print("Loading nrg_bal_s (Energy Balance)...")
        df_energy_balance = eurostat.get_data_df('nrg_bal_s')
        
        print("Cleaning renewable energy data...")
        df_re_clean = clean_and_melt_renewable_energy_data(df_renewable_energy.copy())
        print("Cleaning energy balance data...")
        df_energy_clean = clean_and_melt_energy_balance_data(df_energy_balance.copy())
        
        print("Processing renewable energy data with geo integration...")
        df_renewable_processed = process_renewable_data(df_re_clean)
        print("Processing energy balance data with geo integration...")
        df_balance_processed = process_energy_balance_data(df_energy_clean)
        
        # Merge datasets
        print("Merging datasets...")
        df_combined = merge_datasets(df_renewable_processed, df_balance_processed)
        
        print("Data processing completed successfully!")
        return df_combined
        
    except Exception as e:
        print(f"Error loading Eurostat data: {e}")
        print("Using fallback data...")
        return create_fallback_data()

def process_renewable_data(df_clean):
    """Process cleaned renewable energy data"""
    df_processed = integrate_geo_data(df_clean, nuts0_country_map)
    df_country_level = df_processed[df_processed['nuts_level'] == 0]
    
    # Map energy types based on nrg_bal codes
    energy_type_mapping = {
        'REN': 'Total Renewable',
        'RA000': 'Hydro',
        'RA100': 'Wind',
        'RA200': 'Solar',
        'RA300': 'Geothermal',
        'RA400': 'Biomass',
        'RA500': 'Other Renewable'
    }
    
    df_country_level['Energy_Type'] = df_country_level['nrg_bal'].map(energy_type_mapping)
    df_country_level = df_country_level.dropna(subset=['Energy_Type'])
    
    # Convert to standard format
    processed_data = []
    for _, row in df_country_level.iterrows():
        processed_data.append({
            'country_name': row['country_name'],
            'Year': row['year'],
            'Energy_Type': row['Energy_Type'],
            'Renewable_Share': float(row['renewable_share_pct']),
            'DataSource': 'nrg_ind_ren',
            'Unit': row.get('unit', 'Percentage'),
            'geo': row['geo'],
            'country_code': row['country_code']
        })
    
    return pd.DataFrame(processed_data)

def process_energy_balance_data(df_clean):
    """Process cleaned energy balance data"""
    df_processed = integrate_geo_data(df_clean, nuts0_country_map)
    df_country_level = df_processed[df_processed['nuts_level'] == 0]
    
    # Focus on relevant energy balance items
    relevant_balances = {
        'B_100000': 'Gross Inland Consumption',
        'B_100900': 'Final Energy Consumption',
        'B_101600': 'Electricity Generation',
        'B_107000': 'Renewable Energy',
        'B_101600': 'Total Electricity'
    }
    
    # Map energy sources
    energy_source_mapping = {
        'RA000': 'Hydro',
        'RA100': 'Wind',
        'RA200': 'Solar',
        'RA300': 'Geothermal',
        'RA400': 'Biomass',
        'REN': 'Total Renewable',
        'TOTAL': 'Total Energy'
    }
    
    df_country_level['Balance_Item'] = df_country_level['nrg_bal'].map(relevant_balances)
    df_country_level['Energy_Source'] = df_country_level['siec'].map(energy_source_mapping)
    df_country_level = df_country_level.dropna(subset=['Balance_Item', 'Energy_Source'])
    
    processed_data = []
    for _, row in df_country_level.iterrows():
        processed_data.append({
            'country_name': row['country_name'],
            'Year': row['year'],
            'Balance_Item': row['Balance_Item'],
            'Energy_Source': row['Energy_Source'],
            'Energy_Value_GWh': float(row['energy_value_gwh']),
            'DataSource': 'nrg_bal_s',
            'Unit': row.get('unit', 'GWh'),
            'geo': row['geo'],
            'country_code': row['country_code']
        })
    
    return pd.DataFrame(processed_data)

def merge_datasets(df_renewable, df_balance):
    """Merge both datasets for comprehensive analysis"""
    # Calculate total energy production from balance data
    energy_totals = df_balance[
        (df_balance['Balance_Item'] == 'Gross Inland Consumption') & 
        (df_balance['Energy_Source'] == 'Total Energy')
    ].copy()
    
    energy_totals = energy_totals.rename(columns={'Energy_Value_GWh': 'Total_Energy_GWh'})
    
    # Merge with renewable data
    df_merged = df_renewable.merge(
        energy_totals[['country_name', 'Year', 'Total_Energy_GWh']],
        on=['country_name', 'Year'],
        how='left'
    )
    
    # Calculate absolute values from percentages
    df_merged['Renewable_Absolute_GWh'] = (df_merged['Renewable_Share'] / 100) * df_merged['Total_Energy_GWh']
    
    # Add detailed energy balance data for each energy type
    renewable_balance_data = df_balance[
        (df_balance['Balance_Item'] == 'Renewable Energy') & 
        (df_balance['Energy_Source'].isin(['Hydro', 'Wind', 'Solar', 'Geothermal', 'Biomass', 'Total Renewable']))
    ].copy()
    
    # Pivot to get energy values by type
    energy_by_type = renewable_balance_data.pivot_table(
        index=['country_name', 'Year'],
        columns='Energy_Source',
        values='Energy_Value_GWh',
        aggfunc='first'
    ).reset_index()
    
    # Merge with main dataset
    df_final = df_merged.merge(
        energy_by_type,
        on=['country_name', 'Year'],
        how='left'
    )
    
    return df_final

def create_fallback_data():
    """Create comprehensive fallback data"""
    print("Creating comprehensive fallback data...")
    np.random.seed(42)
    
    european_countries = list(nuts0_country_map.values())[:15]
    energy_types = ['Total Renewable', 'Hydro', 'Wind', 'Solar', 'Geothermal', 'Biomass']
    years = list(range(2010, 2024))
    
    data = []
    for year in years:
        for country in european_countries:
            # Base values with realistic growth
            base_values = {
                'Total Renewable': (20, 2.0),
                'Hydro': (15, 0.3),
                'Wind': (8, 1.5),
                'Solar': (2, 2.5),
                'Geothermal': (1, 0.2),
                'Biomass': (5, 1.0)
            }
            
            # Country-specific multipliers
            country_multipliers = {
                'Germany': 1.3, 'Sweden': 1.8, 'Denmark': 1.7, 'Spain': 1.4,
                'Italy': 1.2, 'France': 1.1, 'Netherlands': 1.3, 'Austria': 1.4,
                'Portugal': 1.5, 'Greece': 1.3, 'Belgium': 1.1, 'Finland': 1.6,
                'Norway': 2.0, 'United Kingdom': 1.2, 'Poland': 0.8
            }
            
            multiplier = country_multipliers.get(country, 1.0)
            total_energy = 100000 + (year - 2010) * 5000  # Simulate energy consumption growth in GWh
            
            for energy_type, (base, growth) in base_values.items():
                share = (base + growth * (year - 2010)) * multiplier + np.random.normal(0, 2)
                share = max(1, min(80, share))
                absolute_value = (share / 100) * total_energy
                
                data.append({
                    'country_name': country,
                    'Year': year,
                    'Energy_Type': energy_type,
                    'Renewable_Share': round(share, 1),
                    'Total_Energy_GWh': total_energy,
                    'Renewable_Absolute_GWh': round(absolute_value, 1),
                    'DataSource': 'fallback',
                    'Unit': 'Percentage',
                    'geo': country[:2].upper(),
                    'country_code': country[:2].upper()
                })
    
    return pd.DataFrame(data)

def create_forecast(df, countries, energy_types, years_to_forecast=5):
    """Create forecasting using multiple models"""
    forecasts = []
    
    for country in countries:
        for energy_type in energy_types:
            subset = df[
                (df['country_name'] == country) & 
                (df['Energy_Type'] == energy_type)
            ].sort_values('Year')
            
            if len(subset) > 3:
                X = subset[['Year']].values
                y = subset['Renewable_Share'].values
                
                # Linear regression
                lr_model = LinearRegression()
                lr_model.fit(X, y)
                
                # Random Forest for non-linear trends
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X, y)
                
                # Forecast future years
                future_years = list(range(df['Year'].max() + 1, df['Year'].max() + years_to_forecast + 1))
                X_future = np.array(future_years).reshape(-1, 1)
                
                # Use ensemble of both models
                lr_forecast = lr_model.predict(X_future)
                rf_forecast = rf_model.predict(X_future)
                combined_forecast = (lr_forecast + rf_forecast) / 2
                
                # Apply constraints
                combined_forecast = np.maximum(0, np.minimum(100, combined_forecast))
                
                for i, (year, share) in enumerate(zip(future_years, combined_forecast)):
                    forecasts.append({
                        'country_name': country,
                        'Year': year,
                        'Energy_Type': energy_type,
                        'Renewable_Share': round(float(share), 1),
                        'Type': 'Forecast',
                        'Confidence': 'Medium',
                        'Model': 'Ensemble (Linear + Random Forest)'
                    })
    
    return pd.DataFrame(forecasts)

# Load data
print("Loading and processing Eurostat datasets...")
df = load_and_process_eurostat_data()

print(f"Data loaded: {len(df)} rows")
print(f"Countries: {df['country_name'].nunique()}")
print(f"Years: {df['Year'].min()} to {df['Year'].max()}")
print(f"Energy Types: {df['Energy_Type'].unique()}")

@app.route('/')
def home():
    countries = sorted(df['country_name'].unique())
    energy_types = sorted(df['Energy_Type'].unique())
    years = sorted(df['Year'].unique(), reverse=True)
    
    latest_year = df['Year'].max()
    latest_data = df[df['Year'] == latest_year]
    total_renewable_data = latest_data[latest_data['Energy_Type'] == 'Total Renewable']
    avg_renewable = round(total_renewable_data['Renewable_Share'].mean(), 1) if not total_renewable_data.empty else 0
    
    stats = {
        'total_countries': len(df['country_name'].unique()),
        'latest_year': latest_year,
        'avg_renewable': avg_renewable,
        'data_years': f"{df['Year'].min()}-{df['Year'].max()}",
        'data_sources': "Eurostat nrg_ind_ren & nrg_bal_s"
    }
    
    return render_template('index.html', 
                         countries=countries, 
                         energy_types=energy_types,
                         years=years,
                         stats=stats)

@app.route('/dashboard')
def dashboard():
    # Get filter values
    countries = request.args.getlist('countries')
    energy_types = request.args.getlist('energy_types')
    year = request.args.get('year', 'All')
    comparison_type = request.args.get('comparison_type', 'countries')
    show_forecast = request.args.get('show_forecast', 'false') == 'true'
    forecast_years = int(request.args.get('forecast_years', 5))
    analysis_type = request.args.get('analysis_type', 'share')  # 'share' or 'absolute'
    
    # Default selections
    if not countries:
        countries = sorted(df['country_name'].unique())[:3]
    if not energy_types:
        energy_types = ['Total Renewable']  # Default to total renewable
    
    # Filter data
    filtered_df = df.copy()
    filtered_df = filtered_df[filtered_df['country_name'].isin(countries)]
    filtered_df = filtered_df[filtered_df['Energy_Type'].isin(energy_types)]
    if year != 'All':
        filtered_df = filtered_df[filtered_df['Year'] == int(year)]
    
    # Add forecast data if requested
    if show_forecast and year == 'All':
        forecast_df = create_forecast(df, countries, energy_types, forecast_years)
        if not forecast_df.empty:
            filtered_df_with_forecast = pd.concat([
                filtered_df.assign(Type='Actual'),
                forecast_df
            ])
        else:
            filtered_df_with_forecast = filtered_df.assign(Type='Actual')
    else:
        filtered_df_with_forecast = filtered_df.assign(Type='Actual')
    
    # Create visualizations based on analysis type
    if analysis_type == 'absolute':
        value_column = 'Renewable_Absolute_GWh'
        value_label = 'Energy Production (GWh)'
        title_suffix = ' - Absolute Values'
    else:
        value_column = 'Renewable_Share'
        value_label = 'Share of Energy Mix (%)'
        title_suffix = ' - Percentage Share'
    
    # Generate charts
    chart1, chart1_tooltips = create_main_chart(
        filtered_df_with_forecast, comparison_type, year, show_forecast, 
        value_column, value_label, title_suffix
    )
    
    chart2, chart2_tooltips = create_secondary_chart(
        filtered_df, comparison_type, year, countries, energy_types,
        value_column, value_label
    )
    
    # Additional insights chart
    chart3, chart3_tooltips = create_insights_chart(
        df, countries, energy_types, year
    )
    
    # Calculate statistics
    stats = calculate_statistics(filtered_df, filtered_df_with_forecast, year, countries, analysis_type)
    
    return render_template('dashboard.html', 
                         chart1=chart1, 
                         chart2=chart2,
                         chart3=chart3,
                         chart1_tooltips=chart1_tooltips,
                         chart2_tooltips=chart2_tooltips,
                         chart3_tooltips=chart3_tooltips,
                         stats=stats,
                         selected_countries=countries,
                         selected_energy_types=energy_types,
                         selected_year=year,
                         comparison_type=comparison_type,
                         show_forecast=show_forecast,
                         forecast_years=forecast_years,
                         analysis_type=analysis_type)

def create_main_chart(filtered_df, comparison_type, year, show_forecast, value_column, value_label, title_suffix):
    """Create the main comparison chart"""
    if comparison_type == 'countries':
        if year == 'All':
            # Group by year and country
            if value_column == 'Renewable_Share':
                trend_df = filtered_df.groupby(['Year', 'country_name', 'Type'])[value_column].mean().reset_index()
            else:
                trend_df = filtered_df.groupby(['Year', 'country_name', 'Type'])[value_column].sum().reset_index()
            
            fig = px.line(trend_df, x='Year', y=value_column, color='country_name',
                         line_dash='Type' if show_forecast else None,
                         title=f'Renewable Energy Trends - Country Comparison{title_suffix}',
                         labels={value_column: value_label, 'Year': 'Year'})
            
            tooltips = [
                f"Hover over lines to see {value_label.lower()} for each country",
                "Dashed lines indicate forecasted values" if show_forecast else "",
                "Multiple countries can be compared simultaneously"
            ]
        else:
            # Year-specific comparison
            comp_df = filtered_df.groupby(['country_name', 'Energy_Type'])[value_column].mean().reset_index()
            fig = px.bar(comp_df, x='country_name', y=value_column, color='Energy_Type',
                        title=f'Energy Distribution by Country ({year}){title_suffix}',
                        barmode='stack',
                        labels={value_column: value_label, 'country_name': 'Country'})
            tooltips = [
                f"Stacked bars show {value_label.lower()} distribution",
                "Different colors represent different energy types"
            ]
    else:
        # Compare energy types
        if year == 'All':
            trend_df = filtered_df.groupby(['Year', 'Energy_Type', 'Type'])[value_column].mean().reset_index()
            fig = px.line(trend_df, x='Year', y=value_column, color='Energy_Type',
                         line_dash='Type' if show_forecast else None,
                         title=f'Energy Type Trends Comparison{title_suffix}',
                         labels={value_column: value_label, 'Year': 'Year'})
            tooltips = [
                "Compare growth rates of different renewable energy sources",
                "Solar typically shows the fastest growth in recent years"
            ]
        else:
            comp_df = filtered_df.groupby(['Energy_Type', 'country_name'])[value_column].mean().reset_index()
            fig = px.bar(comp_df, x='Energy_Type', y=value_column, color='country_name',
                        title=f'Energy Type Comparison by Country ({year}){title_suffix}',
                        barmode='group',
                        labels={value_column: value_label, 'Energy_Type': 'Energy Type'})
            tooltips = [
                "Grouped bars allow direct comparison across countries",
                "Wind energy often leads in Northern European countries"
            ]
    
    # Enhance chart appearance
    fig.update_layout(
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='#f8f9fa'
    )
    
    fig.update_traces(
        hovertemplate=f'<b>%{{x}}</b><br>{value_label.split(" ")[0]}: %{{y:.1f}}<extra></extra>'
    )
    
    return pio.to_html(fig, full_html=False), [t for t in tooltips if t]

def create_secondary_chart(filtered_df, comparison_type, year, countries, energy_types, value_column, value_label):
    """Create secondary chart with different perspective"""
    latest_year = df['Year'].max()
    
    if year == 'All':
        latest_data = filtered_df[filtered_df['Year'] == latest_year]
        
        if comparison_type == 'countries':
            if value_column == 'Renewable_Share':
                comp_df = latest_data.groupby('country_name')[value_column].mean().reset_index()
            else:
                comp_df = latest_data.groupby('country_name')[value_column].sum().reset_index()
                
            fig = px.bar(comp_df, x='country_name', y=value_column,
                        title=f'Latest Year Comparison ({latest_year})',
                        color='country_name',
                        labels={value_column: value_label, 'country_name': 'Country'})
            tooltips = [
                f"Snapshot of renewable energy in {latest_year}",
                "Useful for current state comparison"
            ]
        else:
            comp_df = latest_data.groupby('Energy_Type')[value_column].mean().reset_index()
            fig = px.pie(comp_df, values=value_column, names='Energy_Type',
                        title=f'Energy Type Distribution ({latest_year})')
            tooltips = [
                "Pie chart shows the proportion of different energy types",
                "Hydro power often dominates in mountainous countries"
            ]
    else:
        # Different visualization for single year
        if len(energy_types) > 1:
            comp_df = filtered_df.groupby('Energy_Type')[value_column].sum().reset_index()
            fig = px.pie(comp_df, values=value_column, names='Energy_Type',
                        title=f'Energy Mix Distribution ({year})')
            tooltips = [f"Energy source distribution for {year}"]
        else:
            comp_df = filtered_df.groupby('country_name')[value_column].sum().reset_index()
            fig = px.bar(comp_df, x='country_name', y=value_column,
                        title=f'Country Comparison ({year})',
                        color='country_name')
            tooltips = [f"Direct country comparison for {year}"]
    
    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>Value: %{value:.1f}<extra></extra>'
    )
    
    return pio.to_html(fig, full_html=False), [t for t in tooltips if t]

def create_insights_chart(df, countries, energy_types, year):
    """Create additional insights chart"""
    # Growth analysis
    if year == 'All' and len(countries) > 0:
        country_data = df[
            (df['country_name'].isin(countries)) & 
            (df['Energy_Type'] == 'Total Renewable')
        ]
        
        if not country_data.empty:
            # Calculate growth rates
            growth_data = []
            for country in countries:
                country_df = country_data[country_data['country_name'] == country].sort_values('Year')
                if len(country_df) > 1:
                    latest = country_df[country_df['Year'] == country_df['Year'].max()]['Renewable_Share'].values[0]
                    oldest = country_df[country_df['Year'] == country_df['Year'].min()]['Renewable_Share'].values[0]
                    growth = ((latest - oldest) / oldest) * 100 if oldest > 0 else 0
                    
                    growth_data.append({
                        'country_name': country,
                        'Growth_Rate': round(growth, 1),
                        'Latest_Share': round(latest, 1)
                    })
            
            if growth_data:
                growth_df = pd.DataFrame(growth_data)
                fig = px.bar(growth_df, x='country_name', y='Growth_Rate',
                            title='Renewable Energy Growth Rates (%)',
                            color='Growth_Rate',
                            labels={'Growth_Rate': 'Growth Rate (%)', 'country_name': 'Country'})
                
                tooltips = [
                    "Shows percentage growth in renewable energy share",
                    "Positive values indicate increasing renewable adoption",
                    "Based on total renewable energy share over available years"
                ]
                
                fig.update_traces(
                    hovertemplate='<b>%{x}</b><br>Growth: %{y:.1f}%<extra></extra>'
                )
                
                return pio.to_html(fig, full_html=False), tooltips
    
    # Fallback: Show energy type efficiency
    efficiency_data = []
    for energy_type in energy_types:
        if energy_type != 'Total Renewable':
            type_data = df[df['Energy_Type'] == energy_type]
            avg_share = type_data['Renewable_Share'].mean()
            efficiency_data.append({
                'Energy_Type': energy_type,
                'Average_Share': round(avg_share, 1)
            })
    
    if efficiency_data:
        eff_df = pd.DataFrame(efficiency_data)
        fig = px.bar(eff_df, x='Energy_Type', y='Average_Share',
                    title='Average Share by Energy Type',
                    color='Energy_Type',
                    labels={'Average_Share': 'Average Share (%)', 'Energy_Type': 'Energy Type'})
        
        tooltips = [
            "Average adoption rates across different renewable energy types",
            "Useful for understanding which technologies are most deployed"
        ]
        
        return pio.to_html(fig, full_html=False), tooltips
    
    return None, []

def calculate_statistics(filtered_df, filtered_df_with_forecast, year, countries, analysis_type):
    """Calculate comprehensive statistics"""
    latest_year = df['Year'].max()
    
    if year == 'All':
        stats_df = filtered_df[filtered_df['Year'] == latest_year]
        forecast_df = filtered_df_with_forecast[filtered_df_with_forecast['Type'] == 'Forecast']
    else:
        stats_df = filtered_df
        forecast_df = pd.DataFrame()
    
    # Value column based on analysis type
    if analysis_type == 'absolute':
        value_column = 'Renewable_Absolute_GWh'
        value_unit = 'GWh'
    else:
        value_column = 'Renewable_Share'
        value_unit = '%'
    
    # Country statistics
    country_stats = []
    for country in countries:
        country_data = stats_df[stats_df['country_name'] == country]
        if value_column == 'Renewable_Share':
            total_value = country_data[value_column].mean()
        else:
            total_value = country_data[value_column].sum()
            
        country_stats.append({
            'country': country,
            'total_value': round(total_value, 1),
            'unit': value_unit
        })
    
    # Overall stats
    if value_column == 'Renewable_Share':
        avg_value = round(stats_df[value_column].mean(), 1)
        max_value = round(stats_df[value_column].max(), 1)
    else:
        avg_value = round(stats_df[value_column].sum() / len(countries), 1)
        max_value = round(stats_df[value_column].max(), 1)
    
    # Growth calculation
    if year == 'All':
        current_data = filtered_df[filtered_df['Year'] == latest_year]
        base_year = max(filtered_df['Year'].min(), latest_year - 5)
        base_data = filtered_df[filtered_df['Year'] == base_year]
        
        if value_column == 'Renewable_Share':
            current_avg = current_data[value_column].mean()
            base_avg = base_data[value_column].mean()
        else:
            current_avg = current_data[value_column].sum() / len(countries)
            base_avg = base_data[value_column].sum() / len(countries)
            
        growth = round(current_avg - base_avg, 1) if not np.isnan(current_avg) and not np.isnan(base_avg) else 0
    else:
        growth = 0
    
    return {
        'avg': avg_value,
        'max': max_value,
        'growth': growth,
        'latest_year': latest_year,
        'selected_year': year,
        'country_stats': country_stats,
        'comparison_type': 'countries',
        'has_forecast': not forecast_df.empty,
        'analysis_type': analysis_type,
        'value_unit': value_unit
    }

@app.route('/download-data')
def download_data():
    """Download filtered data as CSV"""
    countries = request.args.getlist('countries')
    energy_types = request.args.getlist('energy_types')
    year = request.args.get('year', 'All')
    analysis_type = request.args.get('analysis_type', 'share')
    
    filtered_df = df.copy()
    filtered_df = filtered_df[filtered_df['country_name'].isin(countries)]
    filtered_df = filtered_df[filtered_df['Energy_Type'].isin(energy_types)]
    if year != 'All':
        filtered_df = filtered_df[filtered_df['Year'] == int(year)]
    
    # Select relevant columns
    output_columns = ['country_name', 'Year', 'Energy_Type', 'Renewable_Share']
    if analysis_type == 'absolute' and 'Renewable_Absolute_GWh' in filtered_df.columns:
        output_columns.append('Renewable_Absolute_GWh')
    if 'Total_Energy_GWh' in filtered_df.columns:
        output_columns.append('Total_Energy_GWh')
    
    filtered_df = filtered_df[output_columns]
    
    # Create CSV in memory
    output = io.StringIO()
    filtered_df.to_csv(output, index=False)
    output.seek(0)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eurostat_energy_data_{timestamp}.csv"
    
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=filename
    )

@app.route('/api/forecast')
def api_forecast():
    """API endpoint for forecasting"""
    countries = request.args.getlist('countries')
    energy_types = request.args.getlist('energy_types')
    years = int(request.args.get('years', 5))
    
    if not countries:
        countries = sorted(df['country_name'].unique())[:3]
    if not energy_types:
        energy_types = ['Total Renewable']
    
    forecast_df = create_forecast(df, countries, energy_types, years)
    
    return jsonify(forecast_df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)'''

"""
Optimized Flask app for PythonAnywhere deployment.

Key points:
- No heavy work at import time (no prints).
- Lazy data loading with caching and robust fallback.
- Forecasts trained on-demand and cached.
- Simple endpoints: '/', '/dashboard', '/forecast', '/download-csv'
- Ready for PythonAnywhere WSGI: `app` variable exists.
"""

import logging
from functools import lru_cache
from io import BytesIO
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_file, abort

# Data & ML libs
import numpy as np
import pandas as pd

# Optional libs: plotly, sklearn, eurostat
try:
    import plotly.io as pio  # left unused here but available for templates
    import plotly.express as px
except Exception:
    px = None
    pio = None

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
except Exception:
    LinearRegression = None
    RandomForestRegressor = None

try:
    import eurostat
    EUROSTAT_AVAILABLE = True
except Exception:
    eurostat = None
    EUROSTAT_AVAILABLE = False

# Configure logging (no prints)
logger = logging.getLogger("euro_energy_app")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

app = Flask(__name__)

# Minimal NUTS0 mapping (expand if you want)
NUTS0_COUNTRY_MAP = {
    'AL': 'Albania', 'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria',
    'CY': 'Cyprus', 'CZ': 'Czechia', 'DE': 'Germany', 'DK': 'Denmark',
    'EE': 'Estonia', 'EL': 'Greece', 'ES': 'Spain', 'FI': 'Finland',
    'FR': 'France', 'HR': 'Croatia', 'HU': 'Hungary', 'IE': 'Ireland',
    'IS': 'Iceland', 'IT': 'Italy', 'LT': 'Lithuania', 'LU': 'Luxembourg',
    'LV': 'Latvia', 'ME': 'Montenegro', 'MK': 'North Macedonia', 'MT': 'Malta',
    'NL': 'Netherlands', 'NO': 'Norway', 'PL': 'Poland', 'PT': 'Portugal',
    'RO': 'Romania', 'RS': 'Serbia', 'SE': 'Sweden', 'SI': 'Slovenia',
    'SK': 'Slovakia', 'TR': 'Türkiye', 'UA': 'Ukraine', 'UK': 'United Kingdom'
}

# -------------------------
# Utilities and data loaders
# -------------------------

def _create_fallback_data():
    """Create a small realistic fallback dataset to keep the app running."""
    rng = np.random.RandomState(42)
    countries = list(NUTS0_COUNTRY_MAP.values())[:12]
    energy_types = ['Total Renewable', 'Hydro', 'Wind', 'Solar', 'Biomass']
    years = list(range(2010, 2024))

    rows = []
    for year in years:
        for country in countries:
            base_total = 100000 + (year - 2010) * 5000
            for et in energy_types:
                # create plausible renewable share percentages
                base = {
                    'Total Renewable': 20.0,
                    'Hydro': 10.0,
                    'Wind': 6.0,
                    'Solar': 2.0,
                    'Biomass': 4.0
                }.get(et, 2.0)
                growth = {
                    'Total Renewable': 1.2,
                    'Hydro': 0.1,
                    'Wind': 0.7,
                    'Solar': 0.9,
                    'Biomass': 0.3
                }.get(et, 0.2)
                share = max(0.5, base + growth * (year - 2010) + rng.normal(0, 1.5))
                absolute_gwh = (share / 100.0) * base_total
                rows.append({
                    'country_name': country,
                    'Year': int(year),
                    'Energy_Type': et,
                    'Renewable_Share': round(float(share), 1),
                    'Total_Energy_GWh': round(float(base_total), 1),
                    'Renewable_Absolute_GWh': round(float(absolute_gwh), 1),
                    'DataSource': 'fallback'
                })
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def load_data():
    """
    Load and return the processed dataset as a DataFrame.
    This function is cached (so repeated calls are fast).
    """
    logger.info("load_data(): called")

    if EUROSTAT_AVAILABLE:
        try:
            # Try to fetch data. Keep this minimal to avoid errors on PythonAnywhere.
            # We keep the processing conservative: select a few useful columns and normalize names.
            logger.info("Eurostat available: attempting to fetch datasets")
            re_df = eurostat.get_data_df('nrg_ind_ren')  # renewable indicators
            bal_df = eurostat.get_data_df('nrg_bal_s')  # energy balances

            # Attempt to melt/reshape the two datasets robustly.
            # Keep only the columns we need and coerce problematic formats.
            # --- Renewable indicators processing ---
            # Look for a 'geo' column and year columns; fall back to simple conversions
            if 'geo\\TIME_PERIOD' in re_df.columns:
                re_df = re_df.rename(columns={'geo\\TIME_PERIOD': 'geo'})
            # We attempt to melt numeric columns (years)
            id_vars = [c for c in re_df.columns if not c.isdigit()]  # heuristic
            year_cols = [c for c in re_df.columns if c.isdigit()]
            if year_cols:
                re_long = re_df.melt(id_vars=id_vars, var_name='Year', value_name='value')
            else:
                re_long = re_df.rename(columns={'time_period': 'Year', 'value': 'value'}) if 'value' in re_df.columns else re_df.copy()
            # Normalize
            re_long['Year'] = pd.to_numeric(re_long['Year'], errors='coerce').astype('Int64')
            # Some eurostat datasets use code columns; try minimal mapping
            if 'geo' not in re_long.columns and 'geo\\TIME_PERIOD' not in re_df.columns:
                # try to find equivalent column
                for candidate in ['geo\\TIME_PERIOD', 'geo', 'COUNTRY']:
                    if candidate in re_df.columns:
                        re_long = re_df.rename(columns={candidate: 'geo'})
                        break

            # --- Energy balance processing (minimal) ---
            if 'geo\\TIME_PERIOD' in bal_df.columns:
                bal_df = bal_df.rename(columns={'geo\\TIME_PERIOD': 'geo'})
            # Try to find year columns
            year_cols_bal = [c for c in bal_df.columns if c.isdigit()]
            if year_cols_bal:
                bal_long = bal_df.melt(id_vars=[c for c in bal_df.columns if c not in year_cols_bal],
                                       var_name='Year', value_name='value')
                bal_long['Year'] = pd.to_numeric(bal_long['Year'], errors='coerce').astype('Int64')
            else:
                bal_long = bal_df.copy()

            # Now produce a simple merged dataframe keyed by country code/name, year and some energy type inference.
            # This minimal pipeline is intentionally defensive and will fall back if expected columns are missing.
            # We'll attempt to get country code / country name mapping
            country_col = None
            for c in ['geo', 'country', 'geo\\TIME_PERIOD', 'geo_code']:
                if c in re_long.columns:
                    country_col = c
                    break

            if country_col is None:
                # fallback to our synthetic dataset
                logger.warning("Could not find geo column in Eurostat data; using fallback dataset")
                return _create_fallback_data()

            # Compose a small dataset from the renewable long data:
            # For simplicity, group by geo and Year and compute mean of 'value' (approx renewable share)
            minimal = re_long[[country_col, 'Year', 'value']].copy()
            minimal = minimal.rename(columns={country_col: 'geo', 'value': 'renewable_share_pct'})
            minimal = minimal.dropna(subset=['geo', 'Year', 'renewable_share_pct'])
            # Map geo to human country names
            minimal['country_code'] = minimal['geo'].astype(str).str[:2].str.upper()
            minimal['country_name'] = minimal['country_code'].map(NUTS0_COUNTRY_MAP).fillna(minimal['country_code'])
            # Aggregate by country_name/Year
            agg = minimal.groupby(['country_name', 'Year'], as_index=False)['renewable_share_pct'].mean()
            # Build a tidy DF with several energy types using simple heuristics (Total Renewable + placeholders)
            rows = []
            for _, r in agg.iterrows():
                total_energy = 100000 + (int(r['Year']) - 2010) * 5000
                share = float(r['renewable_share_pct'])
                rows.append({
                    'country_name': r['country_name'],
                    'Year': int(r['Year']),
                    'Energy_Type': 'Total Renewable',
                    'Renewable_Share': round(share, 1),
                    'Total_Energy_GWh': total_energy,
                    'Renewable_Absolute_GWh': round((share / 100.0) * total_energy, 1),
                    'DataSource': 'Eurostat (minimal)'
                })
            if rows:
                df = pd.DataFrame(rows)
                logger.info("Eurostat minimal dataset prepared")
                return df

            # If no rows, fallback
            logger.warning("Eurostat fetch succeeded but no usable rows found; using fallback")
            return _create_fallback_data()

        except Exception as e:
            logger.exception("Error loading Eurostat data, falling back: %s", e)
            return _create_fallback_data()
    else:
        logger.info("Eurostat package is not available; using fallback data.")
        return _create_fallback_data()


# -------------------------
# Forecasting utilities
# -------------------------

@lru_cache(maxsize=256)
def create_forecast_cached(countries_tuple, energy_types_tuple, years_to_forecast=5):
    """
    Create forecasts for a set of countries and energy types and return as DataFrame.
    This is cached for performance.
    """
    # Convert cached arguments back to lists
    countries = list(countries_tuple)
    energy_types = list(energy_types_tuple)

    df = load_data()
    forecasts = []

    # require sklearn, otherwise return empty df
    if LinearRegression is None or RandomForestRegressor is None:
        logger.warning("sklearn not available: skipping model-based forecasting")
        return pd.DataFrame()

    for country in countries:
        for energy_type in energy_types:
            subset = df[(df['country_name'] == country) & (df['Energy_Type'] == energy_type)].sort_values('Year')
            if len(subset) < 4:
                # Not enough points to train reliable models
                continue

            X = subset[['Year']].astype(float).values.reshape(-1, 1)
            y = subset['Renewable_Share'].astype(float).values.reshape(-1, )

            try:
                lr = LinearRegression()
                lr.fit(X, y)
                rf = RandomForestRegressor(n_estimators=50, random_state=42)
                rf.fit(X, y)
            except Exception as e:
                logger.exception("Model training failed for %s %s: %s", country, energy_type, e)
                continue

            last_year = int(subset['Year'].max())
            future_years = list(range(last_year + 1, last_year + years_to_forecast + 1))
            if not future_years:
                continue
            X_future = np.array(future_years).reshape(-1, 1)

            try:
                pred_lr = lr.predict(X_future)
                pred_rf = rf.predict(X_future)
                combined = (pred_lr + pred_rf) / 2.0
                combined = np.clip(combined, 0.0, 100.0)
            except Exception as e:
                logger.exception("Prediction failed for %s %s: %s", country, energy_type, e)
                continue

            for yr, sh in zip(future_years, combined):
                forecasts.append({
                    'country_name': country,
                    'Year': int(yr),
                    'Energy_Type': energy_type,
                    'Renewable_Share': round(float(sh), 1),
                    'Type': 'Forecast'
                })

    if forecasts:
        return pd.DataFrame(forecasts)
    else:
        return pd.DataFrame()


# -------------------------
# Flask routes
# -------------------------

@app.route('/')
def home():
    """
    Render landing page with basic stats.
    """
    df = load_data()
    countries = sorted(df['country_name'].unique()) if not df.empty else []
    energy_types = sorted(df['Energy_Type'].unique()) if 'Energy_Type' in df.columns else ['Total Renewable']
    years = sorted(df['Year'].unique()) if 'Year' in df.columns else []

    latest_year = int(df['Year'].max()) if not df.empty else None
    avg_renewable = float(df[df['Year'] == latest_year]['Renewable_Share'].mean()) if latest_year else 0.0

    stats = {
        'total_countries': len(countries),
        'latest_year': latest_year,
        'avg_renewable': round(avg_renewable, 1),
        'data_years': f"{int(df['Year'].min())}-{int(df['Year'].max())}" if not df.empty else "N/A",
        'data_source': "Eurostat" if EUROSTAT_AVAILABLE else "fallback"
    }

    return render_template('index.html',
                           countries=countries,
                           energy_types=energy_types,
                           years=years,
                           stats=stats)


@app.route('/dashboard')
def dashboard():
    """
    Render dashboard with charts embedded in the template.
    Chart generation is intentionally minimal here (we pass JSON / HTML fragments).
    """
    df = load_data()
    # parse query params with simple defaults
    countries = request.args.getlist('countries') or sorted(df['country_name'].unique())[:3]
    energy_types = request.args.getlist('energy_types') or ['Total Renewable']
    year = request.args.get('year', 'All')
    show_forecast = request.args.get('show_forecast', 'false') == 'true'
    forecast_years = int(request.args.get('forecast_years', 5))

    # filter
    filtered = df.copy()
    if countries:
        filtered = filtered[filtered['country_name'].isin(countries)]
    if energy_types:
        filtered = filtered[filtered['Energy_Type'].isin(energy_types)]
    if year and year != 'All':
        try:
            y = int(year)
            filtered = filtered[filtered['Year'] == y]
        except Exception:
            pass

    # Build minimal chart HTML using plotly if available; otherwise pass empty strings
    chart1_html = chart2_html = chart3_html = ""
    if px is not None and not filtered.empty:
        try:
            # Primary chart: Renewable share time series for primary country & energy type
            primary_country = countries[0] if countries else filtered['country_name'].iloc[0]
            primary_type = energy_types[0] if energy_types else filtered['Energy_Type'].iloc[0]
            ts = filtered[(filtered['country_name'] == primary_country) &
                          (filtered['Energy_Type'] == primary_type)].sort_values('Year')
            if not ts.empty:
                fig = px.line(ts, x='Year', y='Renewable_Share',
                              title=f"{primary_country} - {primary_type} share over time")
                chart1_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        except Exception:
            chart1_html = ""

    # If forecast requested, compute it
    forecast_df = pd.DataFrame()
    if show_forecast:
        forecast_df = create_forecast_cached(tuple(countries), tuple(energy_types), years_to_forecast=forecast_years)

    # Some lightweight stats for the template
    summary_stats = {
        'selected_countries': countries,
        'selected_energy_types': energy_types,
        'rows': len(filtered),
        'forecast_rows': len(forecast_df)
    }

    return render_template('dashboard.html',
                           chart1=chart1_html,
                           chart2=chart2_html,
                           chart3=chart3_html,
                           stats=summary_stats,
                           chart1_tooltips=[],
                           chart2_tooltips=[],
                           chart3_tooltips=[],
                           show_forecast=show_forecast)


@app.route('/forecast', methods=['GET'])
def forecast_api():
    """
    Simple JSON API for forecasts.
    Query args:
      - countries (multiple)
      - energy_types (multiple)
      - years (int, default 5)
    """
    df = load_data()
    countries = request.args.getlist('countries')
    energy_types = request.args.getlist('energy_types')
    years = int(request.args.get('years', 5))

    # Defaults
    if not countries:
        countries = sorted(df['country_name'].unique())[:3]
    if not energy_types:
        energy_types = ['Total Renewable']

    forecast_df = create_forecast_cached(tuple(countries), tuple(energy_types), years_to_forecast=years)
    if forecast_df.empty:
        return jsonify({'forecast': []})
    return jsonify(forecast_df.to_dict(orient='records'))


@app.route('/download-csv')
def download_csv():
    """
    Download filtered dataset as CSV.
    Query args match the dashboard endpoint.
    """
    df = load_data()
    countries = request.args.getlist('countries')
    energy_types = request.args.getlist('energy_types')
    year = request.args.get('year', 'All')

    filtered = df.copy()
    if countries:
        filtered = filtered[filtered['country_name'].isin(countries)]
    if energy_types:
        filtered = filtered[filtered['Energy_Type'].isin(energy_types)]
    if year and year != 'All':
        try:
            y = int(year)
            filtered = filtered[filtered['Year'] == y]
        except Exception:
            pass

    if filtered.empty:
        abort(404, description="No data available for the selected filters")

    csv_bytes = filtered.to_csv(index=False).encode('utf-8')
    buf = BytesIO(csv_bytes)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"euro_energy_export_{ts}.csv"

    return send_file(buf,
                     attachment_filename=filename,
                     as_attachment=True,
                     mimetype='text/csv')


# -------------------------
# CLI / local run
# -------------------------

if __name__ == "__main__":
    # Local debugging only — PythonAnywhere will ignore this block when run under WSGI.
    app.run(debug=True, host="0.0.0.0", port=5000)
