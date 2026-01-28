import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime, time, timedelta

# --- Configuration ---
st.set_page_config(page_title="B1/K1 Manpower SLA App", layout="wide")

# --- Constants ---
# (Month, Day) tuples for National Holidays
HOLIDAYS = [(1, 26), (5, 1), (8, 15), (10, 2), (11, 1)]

# --- Helper Functions ---

def clean_key(series):
    return series.astype(str).str.strip().str.upper()

def clean_status_string(val):
    if pd.isna(val): return ""
    val = re.sub(r'[^A-Z]', '', str(val).upper())
    return val

def clean_time_cols(df, date_col, time_col):
    def combine(r):
        d = r[date_col]
        t = r[time_col]
        if pd.isna(t): return pd.NaT
        if isinstance(t, datetime): t = t.time()
        if isinstance(t, str):
            t = t.strip()
            for fmt in ["%H:%M:%S", "%I:%M:%S %p", "%I:%M %p", "%H:%M", "%H.%M.%S"]:
                try: return datetime.combine(d.date(), datetime.strptime(t, fmt).time())
                except ValueError: continue
            return pd.NaT
        return datetime.combine(d.date(), t)
    return df.apply(combine, axis=1)

def load_raw_data_smart(uploaded_file):
    try:
        uploaded_file.seek(0)
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, header=None)
        else:
            df = pd.read_excel(uploaded_file, header=None)
            
        header_idx = -1
        for i, row in df.head(100).iterrows():
            row_str = row.astype(str).str.upper().values
            if 'STAFF CODE' in row_str and 'LOGIN TIME' in row_str:
                header_idx = i
                break
        
        if header_idx == -1: return None, "Could not find 'STAFF CODE' in file."

        df.columns = df.iloc[header_idx]
        df = df.iloc[header_idx+1:].reset_index(drop=True)
        df.columns = [str(c).strip().upper() for c in df.columns]

        ffill_cols = ['CITY NAME', 'TRAN DATE', 'CENTRE NAME']
        existing = [c for c in ffill_cols if c in df.columns]
        if existing:
            df[existing] = df[existing].replace(r'^\s*$', np.nan, regex=True).ffill()

        if 'STAFF CODE' in df.columns:
            df = df[df['STAFF CODE'].notna()]
            df = df[df['STAFF CODE'].astype(str).str.upper() != 'STAFF CODE']
            
        return df, None
    except Exception as e:
        return None, str(e)

# --- CORE LOGIC ---
def run_audit_logic(raw_df, centre_df, operator_df, settings):
    
    # 1. Prepare Masters
    centre_df.columns = [str(c).strip() for c in centre_df.columns]
    centre_rename = {
        'Center_Name_MIS': 'CENTRE NAME',
        'city_name': 'CITY NAME',
        'Minimum_Counters_to_be_deployed': 'RFP_TARGET',
        'Is_Eligible_Centre': 'IS_ELIGIBLE_CENTRE'
    }
    centre_df = centre_df.rename(columns=centre_rename)

    operator_df.columns = [str(c).strip() for c in operator_df.columns]
    has_status = 'OPR STATUS' in operator_df.columns
    operator_df = operator_df.rename(columns={'OPR CODE': 'STAFF CODE', 'Is_Valid_Operator': 'IS_VALID_OPERATOR'})

    # 2. Data Processing
    raw_df['TRAN DATE'] = pd.to_datetime(raw_df['TRAN DATE'], dayfirst=True, errors='coerce')
    raw_df = raw_df.dropna(subset=['TRAN DATE'])

    min_date = raw_df['TRAN DATE'].min()
    max_date = raw_df['TRAN DATE'].max()
    full_date_range = pd.date_range(start=min_date, end=max_date)
    
    def is_holiday(d): return (d.month, d.day) in HOLIDAYS
    count_working_days_std = sum(1 for d in full_date_range if not is_holiday(d))

    raw_df['Actual_Login_DT'] = clean_time_cols(raw_df, 'TRAN DATE', 'LOGIN TIME')
    raw_df['Actual_Logout_DT'] = clean_time_cols(raw_df, 'TRAN DATE', 'LOGOUT TIME')

    raw_df['Bound_Start'] = raw_df['TRAN DATE'].apply(lambda x: datetime.combine(x.date(), time(8, 0)))
    raw_df['Bound_End'] = raw_df['TRAN DATE'].apply(lambda x: datetime.combine(x.date(), time(19, 0)))
    raw_df['Effective_Login'] = raw_df[['Actual_Login_DT', 'Bound_Start']].max(axis=1)
    raw_df['Effective_Logout'] = raw_df[['Actual_Logout_DT', 'Bound_End']].min(axis=1)
    raw_df['Duration_Minutes'] = (raw_df['Effective_Logout'] - raw_df['Effective_Login']).dt.total_seconds() / 60
    
    # Filter for Positive Duration (Basic Sanity)
    raw_df = raw_df[raw_df['Duration_Minutes'] > 0].copy()

    # 3. Conflict Resolution
    if 'COUNTER NO' not in raw_df.columns: raw_df['COUNTER NO'] = 1
    
    raw_df = raw_df.sort_values(['CITY NAME', 'CENTRE NAME', 'TRAN DATE', 'COUNTER NO', 'Duration_Minutes'], ascending=[True, True, True, True, False])
    raw_df['Is_Duplicate_Counter'] = raw_df.duplicated(['CITY NAME', 'CENTRE NAME', 'TRAN DATE', 'COUNTER NO'], keep='first').astype(int)
    
    raw_df = raw_df.sort_values(['STAFF CODE', 'TRAN DATE', 'Duration_Minutes'], ascending=[True, True, False])
    raw_df['Is_CrossCentre_Duplicate'] = raw_df.duplicated(['STAFF CODE', 'TRAN DATE'], keep='first').astype(int)

    # 4. Merging & Validation
    raw_df['MERGE_KEY'] = clean_key(raw_df['CITY NAME']) + "_" + clean_key(raw_df['CENTRE NAME'])
    centre_df['MERGE_KEY'] = clean_key(centre_df['CITY NAME']) + "_" + clean_key(centre_df['CENTRE NAME'])
    
    master_subset = centre_df[['MERGE_KEY', 'IS_ELIGIBLE_CENTRE', 'RFP_TARGET']].drop_duplicates(subset=['MERGE_KEY'])
    raw_df = raw_df.merge(master_subset, on='MERGE_KEY', how='left')
    
    missing_mask = raw_df['RFP_TARGET'].isna()
    unmatched_centers = raw_df.loc[missing_mask, ['CITY NAME', 'CENTRE NAME']].drop_duplicates()
    
    raw_df['RFP_TARGET'] = raw_df['RFP_TARGET'].fillna(settings['default_target'])
    
    # Bus/Mobile Logic
    bus_mask = raw_df['CENTRE NAME'].astype(str).str.upper().str.contains(r'WHEELS|BUS|MOBILE', regex=True)
    raw_df.loc[bus_mask, 'RFP_TARGET'] = 0
    raw_df.loc[bus_mask, 'IS_ELIGIBLE_CENTRE'] = 0
    
    raw_df['Is_Eligible_Centre'] = pd.to_numeric(raw_df['IS_ELIGIBLE_CENTRE'], errors='coerce').fillna(0).astype(int)
    raw_df['Is_Eligible_Centre'] = np.where(raw_df['Is_Eligible_Centre'] == 1, 1, 0)
    raw_df.loc[raw_df['Is_Eligible_Centre'] == 0, 'RFP_TARGET'] = 0

    raw_df['S_KEY'] = clean_key(raw_df['STAFF CODE'])
    operator_df['S_KEY'] = clean_key(operator_df['STAFF CODE'])
    
    merge_cols = ['S_KEY', 'IS_VALID_OPERATOR']
    if has_status: merge_cols.append('OPR STATUS')
    
    op_subset = operator_df[merge_cols].drop_duplicates(subset=['S_KEY'])
    raw_df = raw_df.merge(op_subset, on='S_KEY', how='left')
    
    if has_status:
        raw_df['STATUS_CLEAN'] = raw_df['OPR STATUS'].apply(clean_status_string)
        raw_df['IS_VALID_OPERATOR'] = pd.to_numeric(raw_df['IS_VALID_OPERATOR'], errors='coerce').fillna(0)
        
        # Check active status for Daily Breakdown columns
        raw_df['Is_Status_Active'] = (raw_df['STATUS_CLEAN'] == 'ACTIVE').astype(int)
        
        raw_df['Is_Eligible_Operator'] = np.where(
            (raw_df['STATUS_CLEAN'] == 'ACTIVE') & (raw_df['IS_VALID_OPERATOR'] == 1), 1, 0)
    else:
        raw_df['Is_Status_Active'] = 1 
        raw_df['Is_Eligible_Operator'] = np.where(raw_df['IS_VALID_OPERATOR'] == 1, 1, 0)

    # 5. Final Calculation
    if 'NO OF TRANS' not in raw_df.columns: raw_df['NO OF TRANS'] = 0
    raw_df['Is_Zero_Transaction'] = np.where(raw_df['NO OF TRANS'] == 0, 1, 0)
    raw_df['Is_Min_1hr_Qualified'] = np.where(raw_df['Duration_Minutes'] >= 60, 1, 0)
    
    conditions = (
        (raw_df['Is_Eligible_Centre'] == 1) &
        (raw_df['Is_Eligible_Operator'] == 1) &
        (raw_df['Is_Zero_Transaction'] == 0) &
        (raw_df['Is_Min_1hr_Qualified'] == 1) &
        (raw_df['Is_Duplicate_Counter'] == 0) &
        (raw_df['Is_CrossCentre_Duplicate'] == 0)
    )
    raw_df['Valid_Manpower'] = np.where(conditions, 1, 0)
    raw_df['Valid_Duration'] = raw_df['Duration_Minutes'] * raw_df['Valid_Manpower']

    # 6. Aggregation & Gap Filling
    
    # Flags for Daily Summary (Rejection Reasons)
    raw_df['Total_Raw_Count'] = 1 # Helper to count raw rows
    raw_df['Flag_Zero_Trans'] = raw_df['Is_Zero_Transaction']
    raw_df['Flag_Less_1Hr'] = (raw_df['Is_Min_1hr_Qualified'] == 0).astype(int)
    raw_df['Flag_Inactive'] = (raw_df['Is_Status_Active'] == 0).astype(int)
    raw_df['Flag_Dupe_Counter'] = raw_df['Is_Duplicate_Counter']
    raw_df['Flag_Dupe_Cross'] = raw_df['Is_CrossCentre_Duplicate']

    daily_agg = raw_df.groupby(['CITY NAME', 'CENTRE NAME', 'TRAN DATE']).agg({
        'Total_Raw_Count': 'sum', # Count of all login attempts
        'Valid_Manpower': 'sum',
        'Valid_Duration': 'sum',
        'RFP_TARGET': 'max',
        'Flag_Zero_Trans': 'sum',
        'Flag_Less_1Hr': 'sum',
        'Flag_Inactive': 'sum',
        'Flag_Dupe_Counter': 'sum',
        'Flag_Dupe_Cross': 'sum'
    }).reset_index()
    
    centers = daily_agg[['CITY NAME', 'CENTRE NAME', 'RFP_TARGET']].drop_duplicates()
    centers['key'] = 1
    dates_df = pd.DataFrame({'TRAN DATE': full_date_range, 'key': 1})
    full_grid = pd.merge(centers, dates_df, on='key').drop('key', axis=1)
    
    full_data = pd.merge(full_grid, daily_agg, on=['CITY NAME', 'CENTRE NAME', 'TRAN DATE', 'RFP_TARGET'], how='left')
    
    # Fill NAs
    fill_cols = ['Total_Raw_Count', 'Valid_Manpower', 'Valid_Duration', 'Flag_Zero_Trans', 'Flag_Less_1Hr', 'Flag_Inactive', 'Flag_Dupe_Counter', 'Flag_Dupe_Cross']
    full_data[fill_cols] = full_data[fill_cols].fillna(0)
    
    full_data['Is_Holiday'] = full_data['TRAN DATE'].apply(is_holiday)
    # V36/39/42 STRICT LOGIC: SUNDAYS ARE WORKING DAYS.
    full_data['Is_Working_Day'] = (~full_data['Is_Holiday']).astype(int)

    def calc_metrics(row):
        if row['Is_Working_Day'] == 0: return pd.Series([0, 0])
        if row['RFP_TARGET'] == 0: return pd.Series([0, 0])
        
        deficit = max(0, row['RFP_TARGET'] - row['Valid_Manpower'])
        surplus = max(0, row['Valid_Manpower'] - row['RFP_TARGET'])
        return pd.Series([deficit, surplus])

    full_data[['Deficit', 'Surplus']] = full_data.apply(calc_metrics, axis=1)
    full_data['Penalty_INR'] = full_data['Deficit'] * 1000
    
    # --- CRITICAL VS PERFORMANCE DEFICIT LOGIC ---
    full_data['Critical_Deficit_Count'] = np.where(
        (full_data['Is_Working_Day'] == 1) & (full_data['RFP_TARGET'] > 0) & (full_data['Valid_Manpower'] == 0), 1, 0
    )
    full_data['Performance_Deficit_Count'] = np.where(
        (full_data['Is_Working_Day'] == 1) & (full_data['RFP_TARGET'] > 0) & (full_data['Valid_Manpower'] > 0) & (full_data['Valid_Manpower'] < full_data['RFP_TARGET']), 1, 0
    )
    
    # 7. Reporting
    
    # A. Daily
    daily_table = full_data.copy()
    daily_table['Required Duration (Mins)'] = (daily_table['RFP_TARGET'] * 660).where(daily_table['Is_Working_Day'] == 1, 0)
    daily_table['Uptime_Raw'] = np.where(
        daily_table['Required Duration (Mins)'] > 0,
        (daily_table['Valid_Duration'] / daily_table['Required Duration (Mins)']) * 100, 0
    )
    daily_table['Daily Uptime %'] = daily_table['Uptime_Raw'].apply(lambda x: f"{x:.2f}%")
    
    daily_table = daily_table.rename(columns={
        'CITY NAME': 'City Name', 'CENTRE NAME': 'Center Name', 'TRAN DATE': 'Date',
        'Total_Raw_Count': 'Total Logins (Raw)', # New Column
        'RFP_TARGET': 'Target', 'Valid_Manpower': 'Valid', 'Deficit': 'Deficit',
        'Surplus': 'Surplus', 'Penalty_INR': 'Penalty',
        'Flag_Zero_Trans': 'Zero Trans', 'Flag_Less_1Hr': '< 1 Hr', 
        'Flag_Inactive': 'Inactive/Blocked', 'Flag_Dupe_Counter': 'Dupe (Counter)', 
        'Flag_Dupe_Cross': 'Dupe (Cross)'
    })
    
    # DETAILED DAILY COLUMNS (Updated with Total Raw)
    daily_cols = [
        'City Name', 'Center Name', 'Date', 'Total Logins (Raw)', 'Target', 'Valid', 
        'Deficit', 'Surplus', 'Penalty', 
        'Zero Trans', '< 1 Hr', 'Inactive/Blocked', 'Dupe (Counter)', 'Dupe (Cross)',
        'Daily Uptime %'
    ]
    daily_out = daily_table[daily_cols]

    # B. Monthly
    full_data['Is_Closed'] = ((full_data['Valid_Manpower'] == 0) & (~full_data['Is_Holiday'])).astype(int)
    
    monthly_agg = full_data.groupby(['CITY NAME', 'CENTRE NAME']).agg({
        'RFP_TARGET': 'max',
        'Is_Working_Day': 'sum',
        'Valid_Manpower': 'sum', 
        'Surplus': 'sum',
        'Deficit': 'sum',
        'Penalty_INR': 'sum',
        'Valid_Duration': 'sum',
        'Is_Closed': 'sum',
        'Critical_Deficit_Count': 'sum',
        'Performance_Deficit_Count': 'sum'
    }).reset_index()
    
    monthly_agg['Minimum Logins'] = monthly_agg['RFP_TARGET'] * monthly_agg['Is_Working_Day']
    
    # UPTIME FORMULA: Valid Duration / (Minimum Logins * 660)
    monthly_agg['Required_Total_Minutes'] = monthly_agg['Minimum Logins'] * 660
    
    monthly_agg['Uptime_Raw'] = np.where(
        monthly_agg['Required_Total_Minutes'] > 0, 
        (monthly_agg['Valid_Duration'] / monthly_agg['Required_Total_Minutes']) * 100, 
        0
    )
    monthly_agg['Center Uptime %'] = monthly_agg['Uptime_Raw'].apply(lambda x: f"{x:.2f}%")
    
    # RENAME COLUMNS
    monthly_summary = monthly_agg.rename(columns={
        'CITY NAME': 'City Name', 
        'CENTRE NAME': 'Center Name',
        'Minimum Logins': 'Minimum Logins as per RFP', 
        'Deficit': 'Deficit Logins',                   
        'Penalty_INR': 'Penalty (INR)',
        'Is_Closed': 'Days Closed (Excl. Holidays)'
    })
    
    # Select requested columns
    monthly_out = monthly_summary[[
        'City Name', 
        'Center Name', 
        'Minimum Logins as per RFP', 
        'Deficit Logins', 
        'Penalty (INR)', 
        'Center Uptime %', 
        'Days Closed (Excl. Holidays)'
    ]]

    diag = {
        'Total Rows': len(raw_df),
        'Unmatched Centers Data': unmatched_centers,
        'Invalid Operator Rows': len(raw_df[raw_df['Is_Eligible_Operator'] == 0]),
        'Working Days Count': count_working_days_std,
        'Raw Uptime Mean': monthly_agg['Uptime_Raw'].mean(), 
        'Unique Centers': monthly_agg['CENTRE NAME'].nunique(),
        'Total Surplus': full_data['Surplus'].sum(),
        'Total Critical': full_data['Critical_Deficit_Count'].sum(),
        'Total Performance': full_data['Performance_Deficit_Count'].sum()
    }
    
    return monthly_out, daily_out, raw_df, diag

# --- UI LAYER ---

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to:", ["📊 Dashboard & Analysis", "📖 Methodology & Rules"])
    
    st.divider()
    st.header("1. Upload Files")
    f_t = st.file_uploader("Raw Data", type=['xlsx', 'csv'])
    f_c = st.file_uploader("Centre Master", type=['xlsx'])
    f_o = st.file_uploader("Operator Details", type=['xlsx'])
    
    st.header("2. Settings")
    default_target = st.number_input("Default Target", min_value=1, value=2)

# --- PAGE: METHODOLOGY ---
if page == "📖 Methodology & Rules":
    st.title("📖 Calculation Methodology & Rules")
    
    st.markdown("### 1. Incident Severity Definitions")
    st.markdown("""
    - 🔴 **Critical Deficit (Service Denial):** A day where a center was **Open** (Working Day) but had **0 Valid Manpower**.
    - 🟡 **Performance Deficit (Shortage):** A day where manpower was present (>0) but less than the Target (e.g., 1 person vs Target 2).
    """)
    
    st.markdown("### 2. Validation Checks (The Gatekeepers)")
    st.info("A login is only considered VALID if it meets ALL the following criteria:")
    st.markdown("""
    1.  **Staff Status:** The Operator's status in the *Operator Master* must be **'ACTIVE'**.
    2.  **Valid Flag:** The Operator's `Is_Valid_Operator` column must be **'1'**.
    3.  **Center Eligibility:** The Center must be marked `Is_Eligible_Centre = 1` in the *Center Master*.
    4.  **Transaction Check:** The login must have **> 0 Transactions**.
    5.  **Duration Check:** The login duration must be **at least 60 minutes**.
    6.  **Exclusions:** Any center with 'WHEELS', 'BUS', or 'MOBILE' in the name is automatically invalidated (Target = 0).
    """)
    
    st.markdown("### 3. Time & Duration Logic")
    st.write("""
    - **Time Clamping:** Work is only counted between **08:00 AM and 07:00 PM**.
    - **Duration Calculation:** `(Effective Logout - Effective Login)` in minutes.
    """)
    
    st.markdown("### 4. Deficit & Penalty Calculation")
    st.markdown("""
    - **Target:** Derived from `Minimum_Counters_to_be_deployed` in Center Master.
    - **Working Days:** Every day of the month is a working day (including Sundays), **EXCEPT** National Holidays.
    - **Daily Deficit:** `Max(0, Target - Valid Manpower)`.
    - **Daily Penalty:** `Deficit * ₹1000`.
    """)
    
    st.markdown("### 5. Monthly Metrics")
    st.write("""
    - **Minimum Logins as per RFP:** `Daily Target * Total Working Days`.
    - **Deficit Logins:** Sum of all daily deficits.
    - **Days Closed:** Count of days (excluding Holidays) where `Valid Manpower == 0`.
    """)
    
    st.markdown("**Center Uptime %:**")
    st.latex(r"\frac{\text{Total Valid Login Duration}}{\text{Minimum Logins (RFP) } \times 660 \text{ mins}}")
    st.markdown("*(This calculates efficiency against the CONTRACTUAL TARGET, not just present staff)*")

# --- PAGE: DASHBOARD ---
elif page == "📊 Dashboard & Analysis":
    st.title("B1/K1 Manpower SLA Calculation App")
    
    if 'results' not in st.session_state:
        st.session_state['results'] = None

    run_clicked = st.button("Run Analysis", type="primary")

    if run_clicked and f_t and f_c and f_o:
        try:
            with st.spinner("Processing Data..."):
                df_t, err = load_raw_data_smart(f_t)
                if err: 
                    st.error(err)
                else:
                    df_c = pd.read_excel(f_c)
                    df_o = pd.read_excel(f_o)
                    
                    settings = {'default_target': default_target}
                    summ, daily, log, diag = run_audit_logic(df_t, df_c, df_o, settings)
                    
                    if summ is not None:
                        st.session_state['results'] = {
                            'summary': summ,
                            'daily': daily,
                            'log': log,
                            'diag': diag
                        }
                        st.success("Analysis Complete!")
                    else:
                        st.error(diag['Error'])
        except Exception as e:
            st.error(f"Execution Error: {e}")

    if st.session_state['results'] is not None:
        res = st.session_state['results']
        summ = res['summary']
        daily = res['daily']
        log = res['log']
        diag = res['diag']

        total_deficit = summ['Deficit Logins'].sum()
        total_critical = diag['Total Critical']
        total_performance = diag['Total Performance']
        total_surplus = diag['Total Surplus']
        avg_uptime = diag['Raw Uptime Mean']
        unique_centers = diag['Unique Centers']
        working_days = diag['Working Days Count']

        st.markdown("### 📊 Dashboard Summary")
        
        # Row 1: Key Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Unique Centers", unique_centers)
        c2.metric("Working Days", working_days)
        c3.metric("Avg Uptime %", f"{avg_uptime:.2f}%")
        c4.metric("Total Deficit Logins", f"{total_deficit:,.0f}")
        
        st.divider()
        
        # Row 2: Severity Split
        st.subheader("⚠️ Deficit Severity Breakdown")
        c5, c6, c7 = st.columns(3)
        c5.metric("🔴 Critical Deficits (Closed)", f"{total_critical}", help="Days where Center was OPEN but 0 Manpower showed up.")
        c6.metric("🟡 Performance Deficits (Short)", f"{total_performance}", help="Days where Manpower was > 0 but < Target.")
        c7.metric("Total Surplus", f"{total_surplus:,.0f}", help="Extra manpower deployed beyond daily target.")

        st.divider()

        missing_data = diag['Unmatched Centers Data']
        if not missing_data.empty:
            real_missing = missing_data[~missing_data['CENTRE NAME'].astype(str).str.upper().str.contains(r'WHEELS|BUS|MOBILE')]
            if not real_missing.empty:
                with st.expander(f"⚠️ Missing Master Data ({len(real_missing)} Centers)"):
                    st.warning(f"Applied Default Target: **{default_target}**")
                    st.dataframe(real_missing, hide_index=True)

        st.subheader("Monthly Summary Report")
        st.dataframe(summ, use_container_width=True)
        
        st.subheader("Daily Performance & Rejection Details")
        st.dataframe(daily, use_container_width=True)
        
        def dl(d): return d.to_csv(index=False).encode('utf-8')
        
        d1, d2, d3 = st.columns(3)
        d1.download_button("📥 Monthly Summary CSV", dl(summ), 'Monthly_Summary.csv')
        d2.download_button("📥 Daily Report CSV", dl(daily), 'Daily_Report.csv')
        d3.download_button("📥 Full Process Log", dl(log), 'Process_Log.csv')

    elif not run_clicked:
        if not (f_t and f_c and f_o):
            st.info("👋 Upload files to start.")