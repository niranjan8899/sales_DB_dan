import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pytz
from io import BytesIO
from statsmodels.tsa.arima.model import ARIMA  # For ARIMA forecasting
from statsmodels.tsa.statespace.sarimax import SARIMAX  # For SARIMA if needed

# Chinese to English month mapping
month_mapping = {
    '一月': 'Jan', '二月': 'Feb', '三月': 'Mar',
    '四月': 'Apr', '五月': 'May', '六月': 'Jun',
    '七月': 'Jul', '八月': 'Aug', '九月': 'Sep',
    '十月': 'Oct', '十一月': 'Nov', '十二月': 'Dec'
}


def load_data(file_path):
    df_dict = pd.read_excel(file_path, sheet_name=None)
    data = []
    for sheet in df_dict:
        if 'Financial Year' not in sheet:
            data.append(df_dict[sheet])
    return pd.concat(data, ignore_index=True)


def preprocess_dates(df):
    df['Issue Date'] = df['Issue Date'].astype(str)
    df['Issue Date'] = df['Issue Date'].str.replace(
        '|'.join(month_mapping.keys()),
        lambda m: month_mapping[m.group()],
        regex=True
    )
    df['Issue Date'] = pd.to_datetime(df['Issue Date'], format='mixed').dt.tz_localize('Australia/Sydney')
    return df


def calculate_fiscal_year(date):
    year = date.year
    if date.month >= 7:
        return f"{year % 100}/{(year + 1) % 100}"
    else:
        return f"{(year - 1) % 100}/{year % 100}"


def calculate_weeks_since_start(df):
    df['Financial Year'] = df['Issue Date'].apply(calculate_fiscal_year)
    df['Branch Start Date'] = df.groupby('Branch Region')['Issue Date'].transform('min')
    df['Weeks Since Start'] = ((df['Issue Date'] - df['Branch Start Date']) / np.timedelta64(1, 'W')).astype(int) + 1
    # Add proper calendar quarters (1-4)
    df['Calendar Quarter'] = df['Issue Date'].dt.quarter
    return df


def main():
    st.set_page_config(page_title="Advanced Sales Dashboard", layout="wide")

    # Load and process data
    df = load_data('New_Formated_Historical and all.xlsx')
    df = preprocess_dates(df)
    df = calculate_weeks_since_start(df)

    # Sidebar configuration
    st.sidebar.header("Filters")
    date_range = st.sidebar.date_input("Select Date Range",
                                       value=[df['Issue Date'].min().date(),
                                              df['Issue Date'].max().date()])

    selected_branches = st.sidebar.multiselect("Select Branches",
                                               df['Branch Region'].unique(),
                                               default=['NSW', 'QLD', 'WA'])

    with st.sidebar.expander("Advanced Filters"):
        fiscal_years = st.multiselect("Filter by Fiscal Year",
                                      df['Financial Year'].unique(),
                                      default=df['Financial Year'].unique().tolist())

        sales_range = st.slider("Sales Range Filter",
                                min_value=int(df['Total'].min()),
                                max_value=int(df['Total'].max()),
                                value=(int(df['Total'].min()), int(df['Total'].max())))

    df_filtered = df[
        (df['Issue Date'].dt.date >= date_range[0]) &
        (df['Issue Date'].dt.date <= date_range[1]) &
        df['Branch Region'].isin(selected_branches) &
        df['Financial Year'].isin(fiscal_years) &
        (df['Total'] >= sales_range[0]) &
        (df['Total'] <= sales_range[1])
        ]

    # Tabs for sections
    tabs = st.tabs(["Branch Comparisons", "Customer Analysis", "Sales Trends", "Export"])

    # BRANCH COMPARISONS TAB
    with tabs[0]:
        st.header("Branch Comparisons")

        # Aggregation selection
        agg_period = st.radio("Select Aggregation Period",
                              ["Weekly", "Monthly", "Quarterly"],
                              index=0,
                              horizontal=True)

        # Forecast parameters
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

        # Handle missing weeks
        if agg_period == "Weekly":
            max_week = df_filtered['Weeks Since Start'].max()
            all_weeks = list(range(1, max_week + forecast_weeks + 1))
            for branch in selected_branches:
                branch_data = comparison_data[comparison_data['Branch Region'] == branch]
                filled_weeks = pd.DataFrame({time_col: all_weeks})
                filled_weeks = filled_weeks.merge(branch_data, on=time_col, how='left')
                filled_weeks.fillna(0, inplace=True)
                comparison_data = pd.concat([comparison_data, filled_weeks])

        # Individual Branch Charts
        for branch in selected_branches:
            branch_data = comparison_data[comparison_data['Branch Region'] == branch]
            fig = px.line(branch_data,
                          x=time_col,
                          y='Total',
                          title=f"{branch} Sales Performance by {period}",
                          labels={time_col: period, 'Total': 'Sales ($)'})
            st.plotly_chart(fig)

        # Combined Comparison Chart
        fig_combined = px.line(comparison_data,
                               x=time_col,
                               y='Total',
                               color='Branch Region',
                               title=f"Combined Sales Comparison by {period}",
                               labels={time_col: period, 'Total': 'Sales ($)'})
        st.plotly_chart(fig_combined)

        # Forecasting Section
        if st.button("Generate Forecast"):
            forecast_data = []
            # Calculate WA's growth rate
            wa_data = df_filtered[df_filtered['Branch Region'] == 'WA']
            wa_sales_series = wa_data.groupby('Weeks Since Start')['Total'].sum()
            wa_growth_rate = wa_sales_series.pct_change().mean()  # Calculate WA's average growth rate

            for branch in selected_branches:
                branch_df = df_filtered[df_filtered['Branch Region'] == branch]

                if model_choice == "Simple Growth":
                    initial_sales = branch_df['Total'].iloc[0]
                    weeks = np.arange(1, forecast_weeks + 1)
                    projected = initial_sales * (1 + growth_rate / 100) ** weeks
                    forecast = pd.DataFrame({
                        'Week': weeks,
                        'Branch': branch,
                        'Projected Sales': projected
                    })
                else:
                    try:
                        sales_series = branch_df['Total'].values
                        model = ARIMA(sales_series, order=(5, 1, 0))
                        model_fit = model.fit()
                        forecast = model_fit.forecast(steps=forecast_weeks)
                        weeks = np.arange(1, forecast_weeks + 1)
                        forecast = pd.DataFrame({
                            'Week': weeks,
                            'Branch': branch,
                            'Projected Sales': forecast
                        })
                    except:
                        st.error("Not enough data for ARIMA forecasting. Try Simple Growth.")
                        continue

                # Apply WA's growth rate as the target
                initial_sales = branch_df.groupby('Weeks Since Start')['Total'].sum().iloc[0]
                weeks = np.arange(1, forecast_weeks + 1)
                projected = initial_sales * (1 + wa_growth_rate) ** weeks  # Using WA growth rate

                forecast = pd.DataFrame({
                    'Week': weeks,
                    'Branch': branch,
                    'Projected Sales (WA Growth)': projected
                })
                forecast_data.append(forecast)

            if forecast_data:
                forecast_df = pd.concat(forecast_data)
                fig_forecast = px.line(forecast_df,
                                       x='Week',
                                       y='Projected Sales (WA Growth)',
                                       color='Branch',
                                       title="Sales Forecast Based on WA Growth Model")
                st.plotly_chart(fig_forecast)

    # CUSTOMER ANALYSIS TAB
    with tabs[1]:
        st.header("Customer Analysis")

        selected_branch = st.selectbox("Select Branch", df['Branch Region'].unique())
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

        # Show only discrepancies
        cust_data_filtered = cust_data[cust_data['Outstanding'] != cust_data['Total']]
        if not cust_data_filtered.empty:
            st.subheader(f"{selected_customer}'s Outstanding Discrepancies")
            st.dataframe(cust_data_filtered[['Issue Date', 'Invoice ID', 'Total', 'Outstanding']],
                         use_container_width=True)

            total_discrepancy = cust_data_filtered['Outstanding'].sum()
            st.warning(f"❗ Total Outstanding Discrepancy: ${total_discrepancy:,.2f}")
        else:
            st.success("No outstanding discrepancies found for this customer.")

        # Quarterly Spend - using Calendar Quarter now
        quarterly = cust_data.groupby(['Financial Year', 'Calendar Quarter'])['Total'].sum().reset_index()
        
        # Map quarter numbers to labels
        quarter_labels = {1: 'Q1 (Jan-Mar)', 2: 'Q2 (Apr-Jun)', 3: 'Q3 (Jul-Sep)', 4: 'Q4 (Oct-Dec)'}
        quarterly['Quarter Label'] = quarterly['Calendar Quarter'].map(quarter_labels)
        
        fig_q = px.bar(quarterly,
                       x='Quarter Label',
                       y='Total',
                       color='Financial Year',
                       title=f"{selected_customer}'s Quarterly Spend Comparison",
                       labels={'Quarter Label': 'Quarter', 'Total': 'Total Spend ($)'})
        fig_q.update_xaxes(categoryorder='array', categoryarray=['Q1 (Jan-Mar)', 'Q2 (Apr-Jun)', 'Q3 (Jul-Sep)', 'Q4 (Oct-Dec)'])
        st.plotly_chart(fig_q)

        # Spend Alert
        spend_threshold = st.slider("Set Spend Drop Alert Threshold (%)", 10, 50, 30)
        cust_data_sorted = cust_data.sort_values('Issue Date')
        cust_data_sorted['12W_Spend'] = cust_data_sorted['Total'].rolling(12).sum()

        latest = cust_data_sorted['12W_Spend'].iloc[-1] if len(cust_data_sorted) >= 12 else 0
        previous = cust_data_sorted['12W_Spend'].iloc[-13] if len(cust_data_sorted) >= 13 else 0
        drop_percent = ((previous - latest) / previous) * 100 if previous != 0 else 0

        st.metric("12-Week Spend Drop", f"{drop_percent:.1f}%")
        if drop_percent < spend_threshold:
            st.toast(f"❗ Spend dropped below {-spend_threshold}% threshold!", icon="⚠️")

    # SALES TRENDS TAB
    with tabs[2]:
        st.header("Sales Trends")

        # Time Series Plot
        st.subheader("Sales Over Time")
        ts_data = df_filtered.groupby(['Issue Date', 'Branch Region'])['Total'].sum().reset_index()
        fig_ts = px.line(ts_data,
                         x='Issue Date',
                         y='Total',
                         color='Branch Region',
                         title="Sales Performance Over Time",
                         labels={"Issue Date": "Date", "Total": "Sales ($)"})
        fig_ts.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig_ts)

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

            fig = px.line(filled_weeks,
                          x='Weeks Since Start',
                          y='Total',
                          title=f"{branch} Weekly Sales Performance",
                          labels={"Weeks Since Start": "Weeks Since Branch Start"})
            st.plotly_chart(fig)

        # Quarterly Sales Comparison
        st.subheader("Quarterly Sales Comparison")
        quarterly_sales = df_filtered.groupby(['Financial Year', 'Calendar Quarter', 'Branch Region'])[
            'Total'].sum().reset_index()
        fig_quarterly = px.bar(quarterly_sales,
                               x='Calendar Quarter',
                               y='Total',
                               color='Branch Region',
                               facet_col='Financial Year',
                               title="Quarterly Sales by Branch and Fiscal Year")
        st.plotly_chart(fig_quarterly)

        # Growth Rate Analysis
        st.subheader("Growth Rate Analysis")
        growth_data = df_filtered.copy()
        growth_data['Growth Rate'] = growth_data.groupby('Branch Region')['Total'].pct_change()
        df_filtered['Growth Rate'] = df_filtered.groupby('Branch Region')[
                                         'Total'].pct_change() * 100  # Percentage change

        correlation_matrix = df_filtered[['Growth Rate', 'Total', 'Calendar Quarter']].corr()

        # Plot the correlation heatmap
        fig_growth = px.imshow(correlation_matrix,
                               labels={'x': 'Metrics', 'y': 'Metrics', 'color': 'Correlation'},
                               title="Correlation Heatmap for Growth and Sales Metrics",
                               color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_growth)

        # Summary Table
        st.subheader("Sales Summary Table")
        filtered_summary = df_filtered.groupby(['Branch Region', 'Customer']).agg(
            Total_Sales=('Total', 'sum'),
            Outstanding_Amount=('Outstanding', 'sum'),
            Invoices=('Invoice ID', 'count')
        ).reset_index()

        # Highlight discrepancies
        filtered_summary['Discrepancy'] = filtered_summary['Total_Sales'] != filtered_summary['Outstanding_Amount']
        styled_df = filtered_summary.style.apply(
            lambda row: ['background-color: yellow' if row['Discrepancy'] else ''],
            subset=['Discrepancy'],
            axis=1
        )

        st.dataframe(styled_df, use_container_width=True)

    # EXPORT TAB
    with tabs[3]:
        st.header("Export Data")

        # CSV Export
        csv_data = df_filtered.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data",
            data=csv_data,
            file_name='filtered_sales.csv',
            mime='text/csv'
        )

        # Plot Export
        if 'fig_ts' in locals():
            fig_bytes = fig_ts.to_image(format="png", width=1000, height=600)
            st.download_button(
                label="Download Time Series Plot",
                data=fig_bytes,
                file_name='time_series_plot.png',
                mime='image/png'
            )


if __name__ == "__main__":
    main()
