import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pytz
from io import BytesIO
import logging
from functools import wraps
import traceback
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chinese to English month mapping
month_mapping = {
    '一月': 'Jan', '二月': 'Feb', '三月': 'Mar',
    '四月': 'Apr', '五月': 'May', '六月': 'Jun',
    '七月': 'Jul', '八月': 'Aug', '九月': 'Sep',
    '十月': 'Oct', '十一月': 'Nov', '十二月': 'Dec'
}

# Error handling decorator
def st_error_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}\n{traceback.format_exc()}")
            st.error(f"An error occurred in {func.__name__}: {str(e)}")
            if "data" in func.__name__.lower():
                st.warning("Please check your data file and try again")
            return None  # or return a default value
    return wrapper

@st_error_handler
def load_data(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at path: {file_path}")
            
        df_dict = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
        if not df_dict:
            raise ValueError("No data found in Excel file")
            
        data = []
        for sheet_name, df in df_dict.items():
            if 'Financial Year' not in sheet_name:
                if df.empty:
                    logger.warning(f"Empty sheet found: {sheet_name}")
                    continue
                data.append(df)
                
        if not data:
            raise ValueError("No valid sheets found in Excel file")
            
        return pd.concat(data, ignore_index=True)
        
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

@st_error_handler
def preprocess_dates(df):
    if df is None or df.empty:
        raise ValueError("Empty DataFrame received for date preprocessing")
        
    try:
        df['Issue Date'] = df['Issue Date'].astype(str)
        df['Issue Date'] = df['Issue Date'].str.replace(
            '|'.join(month_mapping.keys()),
            lambda m: month_mapping[m.group()],
            regex=True
        )
        df['Issue Date'] = pd.to_datetime(df['Issue Date'], format='mixed', errors='coerce').dt.tz_localize('Australia/Sydney')
        
        if df['Issue Date'].isnull().any():
            invalid_dates = df[df['Issue Date'].isnull()]['Issue Date'].unique()
            logger.warning(f"Could not parse dates: {invalid_dates}")
            raise ValueError(f"Could not parse {len(invalid_dates)} dates")
            
        return df
    except Exception as e:
        raise Exception(f"Date preprocessing failed: {str(e)}")

def calculate_fiscal_year(date):
    try:
        year = date.year
        if date.month >= 7:
            return f"{year % 100}/{(year + 1) % 100}"
        return f"{(year - 1) % 100}/{year % 100}"
    except Exception as e:
        logger.error(f"Error calculating fiscal year: {str(e)}")
        return "Unknown"

@st_error_handler
def calculate_weeks_since_start(df):
    if df is None or df.empty:
        raise ValueError("Empty DataFrame received for week calculation")
        
    try:
        df['Financial Year'] = df['Issue Date'].apply(calculate_fiscal_year)
        df['Branch Start Date'] = df.groupby('Branch Region')['Issue Date'].transform('min')
        df['Weeks Since Start'] = ((df['Issue Date'] - df['Branch Start Date']) / np.timedelta64(1, 'W')).astype(int) + 1
        df['Calendar Quarter'] = df['Issue Date'].dt.quarter
        return df
    except Exception as e:
        raise Exception(f"Week calculation failed: {str(e)}")

def safe_plot(fig_func, title="", *args, **kwargs):
    """Wrapper for plotly figures with error handling"""
    try:
        fig = fig_func(*args, **kwargs)
        if title:
            fig.update_layout(title=title)
        st.plotly_chart(fig)
        return fig
    except Exception as e:
        st.error(f"Could not generate visualization: {str(e)}")
        logger.error(f"Plotting error: {str(e)}\n{traceback.format_exc()}")
        return None

def main():
    st.set_page_config(page_title="Advanced Sales Dashboard", layout="wide")
    
    # Initialize session state for error recovery
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    try:
        # Load and process data with error handling
        with st.spinner("Loading data..."):
            if not st.session_state.data_loaded:
                df = load_data('New_Formated_Historical and all.xlsx')
                if df is None:
                    st.error("Failed to load data. Please check the data file.")
                    return
                    
                df = preprocess_dates(df)
                df = calculate_weeks_since_start(df)
                
                if df.empty:
                    st.warning("No data available after processing")
                    return
                
                st.session_state.df = df
                st.session_state.data_loaded = True
            else:
                df = st.session_state.df
                
    except Exception as e:
        st.error(f"Application initialization failed: {str(e)}")
        st.info("Please check if the data file exists and is in the correct format")
        return

    # Sidebar configuration
    with st.sidebar:
        st.header("Filters")
        
        try:
            min_date = df['Issue Date'].min().date()
            max_date = df['Issue Date'].max().date()
            date_range = st.date_input(
                "Select Date Range",
                value=[min_date, max_date],
                min_value=min_date,
                max_value=max_date
            )
            
            if len(date_range) != 2:
                raise ValueError("Please select both start and end dates")
            if date_range[0] > date_range[1]:
                raise ValueError("Start date must be before end date")
                
        except Exception as e:
            st.sidebar.error(f"Date selection error: {str(e)}")
            date_range = [min_date, max_date]
        
        try:
            selected_branches = st.multiselect(
                "Select Branches",
                df['Branch Region'].unique(),
                default=['NSW', 'QLD', 'WA']
            )
            if not selected_branches:
                raise ValueError("Please select at least one branch")
        except Exception as e:
            st.sidebar.error(f"Branch selection error: {str(e)}")
            selected_branches = ['NSW', 'QLD', 'WA']
        
        with st.sidebar.expander("Advanced Filters"):
            try:
                fiscal_years = st.multiselect(
                    "Filter by Fiscal Year",
                    df['Financial Year'].unique(),
                    default=df['Financial Year'].unique().tolist()
                )
                if not fiscal_years:
                    raise ValueError("Please select at least one fiscal year")
            except Exception as e:
                st.error(f"Fiscal year filter error: {str(e)}")
                fiscal_years = df['Financial Year'].unique().tolist()
            
            try:
                sales_range = st.slider(
                    "Sales Range Filter",
                    min_value=int(df['Total'].min()),
                    max_value=int(df['Total'].max()),
                    value=(int(df['Total'].min()), int(df['Total'].max()))
                )
            except Exception as e:
                st.error(f"Sales range error: {str(e)}")
                sales_range = (int(df['Total'].min()), int(df['Total'].max()))

    # Apply filters
    try:
        df_filtered = df[
            (df['Issue Date'].dt.date >= date_range[0]) &
            (df['Issue Date'].dt.date <= date_range[1]) &
            df['Branch Region'].isin(selected_branches) &
            df['Financial Year'].isin(fiscal_years) &
            (df['Total'] >= sales_range[0]) &
            (df['Total'] <= sales_range[1])
        ]
        
        if df_filtered.empty:
            st.warning("No data matches your filters. Showing all data instead.")
            df_filtered = df.copy()
    except Exception as e:
        st.error(f"Filtering error: {str(e)}")
        df_filtered = df.copy()

    # Tabs for sections
    tabs = st.tabs(["Branch Comparisons", "Customer Analysis", "Sales Trends", "Export"])

    # BRANCH COMPARISONS TAB
    with tabs[0]:
        st.header("Branch Comparisons")
        
        try:
            agg_period = st.radio(
                "Select Aggregation Period",
                ["Weekly", "Monthly", "Quarterly"],
                index=0,
                horizontal=True
            )
            
            # Forecasting section
            st.subheader("Sales Forecasting")
            model_choice = st.selectbox("Select Forecasting Model", ["Simple Growth", "ARIMA"])
            
            if model_choice == "Simple Growth":
                growth_rate = st.slider("Growth Rate (%)", 0.0, 20.0, 5.0, 0.5)
            forecast_weeks = st.slider("Forecast Weeks", 1, 52, 12)
            
            # Generate comparison data
            if agg_period == "Weekly":
                time_col = "Weeks Since Start"
                period = "Week"
            elif agg_period == "Monthly":
                df_filtered['Month'] = df_filtered['Issue Date'].dt.to_period('M').astype(str)
                time_col = "Month"
                period = "Month"
            else:
                time_col = "Calendar Quarter"
                period = "Quarter"

            comparison_data = df_filtered.groupby([time_col, 'Branch Region'])['Total'].sum().reset_index()
            
            # Individual Branch Charts
            for branch in selected_branches:
                branch_data = comparison_data[comparison_data['Branch Region'] == branch]
                safe_plot(
                    px.line,
                    title=f"{branch} Sales Performance by {period}",
                    data_frame=branch_data,
                    x=time_col,
                    y='Total',
                    labels={time_col: period, 'Total': 'Sales ($)'}
                )
            
            # Combined Comparison Chart
            safe_plot(
                px.line,
                title=f"Combined Sales Comparison by {period}",
                data_frame=comparison_data,
                x=time_col,
                y='Total',
                color='Branch Region',
                labels={time_col: period, 'Total': 'Sales ($)'}
            )
            
        except Exception as e:
            st.error(f"Error in Branch Comparisons: {str(e)}")

    # CUSTOMER ANALYSIS TAB
    with tabs[1]:
        st.header("Customer Analysis")
        
        try:
            selected_branch = st.selectbox(
                "Select Branch",
                df['Branch Region'].unique()
            )
            
            search_term = st.text_input("Search Customer (ID/Name)", "")
            
            filtered_customers = df[df['Branch Region'] == selected_branch]
            if search_term:
                filtered_customers = filtered_customers[
                    filtered_customers['Customer'].str.contains(search_term, case=False) |
                    filtered_customers['Customer ID'].astype(str).str.contains(search_term, case=False)
                ]
            
            customers = filtered_customers['Customer'].unique()
            selected_customer = st.selectbox("Select Customer", customers)
            
            cust_data = filtered_customers[filtered_customers['Customer'] == selected_customer]
            
            # Show discrepancies
            cust_data_filtered = cust_data[cust_data['Outstanding'] != cust_data['Total']]
            if not cust_data_filtered.empty:
                st.subheader(f"{selected_customer}'s Outstanding Discrepancies")
                st.dataframe(
                    cust_data_filtered[['Issue Date', 'Invoice ID', 'Total', 'Outstanding']],
                    use_container_width=True
                )
                total_discrepancy = cust_data_filtered['Outstanding'].sum()
                st.warning(f"❗ Total Outstanding Discrepancy: ${total_discrepancy:,.2f}")
            else:
                st.success("No outstanding discrepancies found for this customer.")
            
            # Quarterly Spend
            quarterly = cust_data.groupby(['Financial Year', 'Calendar Quarter'])['Total'].sum().reset_index()
            quarter_labels = {1: 'Q1 (Jan-Mar)', 2: 'Q2 (Apr-Jun)', 3: 'Q3 (Jul-Sep)', 4: 'Q4 (Oct-Dec)'}
            quarterly['Quarter Label'] = quarterly['Calendar Quarter'].map(quarter_labels)
            
            fig_q = safe_plot(
                px.bar,
                title=f"{selected_customer}'s Quarterly Spend Comparison",
                data_frame=quarterly,
                x='Quarter Label',
                y='Total',
                color='Financial Year',
                labels={'Quarter Label': 'Quarter', 'Total': 'Total Spend ($)'}
            )
            if fig_q:
                fig_q.update_xaxes(
                    categoryorder='array',
                    categoryarray=['Q1 (Jan-Mar)', 'Q2 (Apr-Jun)', 'Q3 (Jul-Sep)', 'Q4 (Oct-Dec)']
                )
                st.plotly_chart(fig_q)
            
            # Spend Alert
            spend_threshold = st.slider("Set Spend Drop Alert Threshold (%)", 10, 50, 30)
            cust_data_sorted = cust_data.sort_values('Issue Date')
            cust_data_sorted['12W_Spend'] = cust_data_sorted['Total'].rolling(12).sum()
            
            latest = cust_data_sorted['12W_Spend'].iloc[-1] if len(cust_data_sorted) >= 12 else 0
            previous = cust_data_sorted['12W_Spend'].iloc[-13] if len(cust_data_sorted) >= 13 else 0
            drop_percent = ((previous - latest) / previous) * 100 if previous != 0 else 0
            
            st.metric("12-Week Spend Drop", f"{drop_percent:.1f}%")
            if drop_percent > spend_threshold:
                st.toast(f"❗ Spend dropped more than {spend_threshold}%!", icon="⚠️")
                
        except Exception as e:
            st.error(f"Error in Customer Analysis: {str(e)}")

    # SALES TRENDS TAB
    with tabs[2]:
        st.header("Sales Trends")
        
        try:
            # Time Series Plot
            st.subheader("Sales Over Time")
            ts_data = df_filtered.groupby(['Issue Date', 'Branch Region'])['Total'].sum().reset_index()
            safe_plot(
                px.line,
                title="Sales Performance Over Time",
                data_frame=ts_data,
                x='Issue Date',
                y='Total',
                color='Branch Region',
                labels={"Issue Date": "Date", "Total": "Sales ($)"}
            )
            
            # Weekly Sales Comparison
            st.subheader("Weekly Sales Comparison")
            weekly_sales = df_filtered.groupby(['Weeks Since Start', 'Branch Region'])['Total'].sum().reset_index()
            max_week = weekly_sales['Weeks Since Start'].max()
            all_weeks = list(range(1, max_week + 1))
            
            for branch in selected_branches:
                branch_weeks = weekly_sales[weekly_sales['Branch Region'] == branch]
                filled_weeks = pd.DataFrame({'Weeks Since Start': all_weeks})
                filled_weeks = filled_weeks.merge(branch_weeks, on='Weeks Since Start', how='left')
                filled_weeks.fillna(0, inplace=True)
                
                safe_plot(
                    px.line,
                    title=f"{branch} Weekly Sales Performance",
                    data_frame=filled_weeks,
                    x='Weeks Since Start',
                    y='Total',
                    labels={"Weeks Since Start": "Weeks Since Branch Start"}
                )
            
            # Quarterly Sales Comparison
            st.subheader("Quarterly Sales Comparison")
            quarterly_sales = df_filtered.groupby(['Financial Year', 'Calendar Quarter', 'Branch Region'])['Total'].sum().reset_index()
            safe_plot(
                px.bar,
                title="Quarterly Sales by Branch and Fiscal Year",
                data_frame=quarterly_sales,
                x='Calendar Quarter',
                y='Total',
                color='Branch Region',
                facet_col='Financial Year'
            )
            
            # Growth Rate Analysis
            st.subheader("Growth Rate Analysis")
            df_filtered['Growth Rate'] = df_filtered.groupby('Branch Region')['Total'].pct_change() * 100
            correlation_matrix = df_filtered[['Growth Rate', 'Total', 'Calendar Quarter']].corr()
            
            safe_plot(
                px.imshow,
                title="Correlation Heatmap for Growth and Sales Metrics",
                data_frame=correlation_matrix,
                labels={'x': 'Metrics', 'y': 'Metrics', 'color': 'Correlation'},
                color_continuous_scale='RdBu_r'
            )
            
            # Summary Table
            st.subheader("Sales Summary Table")
            try:
                filtered_summary = df_filtered.groupby(['Branch Region', 'Customer']).agg(
                    Total_Sales=('Total', 'sum'),
                    Outstanding_Amount=('Outstanding', 'sum'),
                    Invoices=('Invoice ID', 'count')
                ).reset_index()
                
                filtered_summary['Discrepancy'] = filtered_summary['Total_Sales'] != filtered_summary['Outstanding_Amount']
                st.dataframe(
                    filtered_summary.style.apply(
                        lambda row: ['background-color: yellow' if row['Discrepancy'] else ''],
                        subset=['Discrepancy'],
                        axis=1
                    ),
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Could not generate summary table: {str(e)}")
                
        except Exception as e:
            st.error(f"Error in Sales Trends: {str(e)}")

    # EXPORT TAB
    with tabs[3]:
        st.header("Export Data")
        
        try:
            # CSV Export
            csv_data = df_filtered.to_csv(index=False)
            st.download_button(
                label="Download Filtered Data",
                data=csv_data,
                file_name='filtered_sales.csv',
                mime='text/csv'
            )
            
            # Plot Export (example with time series plot)
            try:
                ts_fig = px.line(
                    df_filtered.groupby(['Issue Date', 'Branch Region'])['Total'].sum().reset_index(),
                    x='Issue Date',
                    y='Total',
                    color='Branch Region',
                    title="Sales Over Time"
                )
                
                plot_bytes = ts_fig.to_image(format="png")
                st.download_button(
                    label="Download Time Series Plot",
                    data=plot_bytes,
                    file_name='time_series_plot.png',
                    mime='image/png'
                )
            except Exception as e:
                st.error(f"Could not generate plot for export: {str(e)}")
                
        except Exception as e:
            st.error(f"Error in Export section: {str(e)}")

if __name__ == "__main__":
    main()
