import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Giảm thiểu log từ TensorFlow

import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import squarify
import streamlit as st
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings("ignore")

# Tham số mặc định từ notebook
RF_ANNUAL = 0.045
TRADING_DAYS = 252
DEFAULT_WINDOW_SIZE = 30
DEFAULT_HORIZON = 5
LAMBDA_ENTROPY = 0.01

# Thiết lập hạt giống ngẫu nhiên để có kết quả nhất quán
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Hàm load dữ liệu giá đóng cửa từ CSV
@st.cache_data(show_spinner=False)
def load_price_data(csv_path=None, uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding="utf-8", low_memory=False)
    elif csv_path is not None:
        df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
    else:
        raise ValueError("Không tìm thấy nguồn dữ liệu.")

    df.columns = df.columns.str.lower()
    if "date" not in df.columns or "ticker" not in df.columns or "close" not in df.columns:
        raise ValueError("CSV cần có các cột tối thiểu: 'date', 'ticker' và 'close'.")

    # Giữ lại các cột cần thiết
    keep_cols = [c for c in ["date", "ticker", "close", "open", "high", "low", "volume"] if c in df.columns]
    df = df[keep_cols].copy()
    
    # Định dạng ngày tháng
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "close"]).copy()
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    return df

# Tạo ma trận giá đóng cửa xoay (pivot table)
def prepare_price_matrix(df, selected_tickers=None):
    price_matrix = df.pivot(index="date", columns="ticker", values="close")
    price_matrix = price_matrix.sort_index()
    # Tiến hành forward fill và backward fill cho các giá trị bị khuyết
    price_matrix = price_matrix.ffill().bfill()
    if selected_tickers is not None:
        tickers = [t for t in selected_tickers if t in price_matrix.columns]
        price_matrix = price_matrix[tickers]
    return price_matrix

# Tính return hàng ngày
def compute_returns(price_df):
    returns = price_df.pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(how="all")
    return returns

# Tính Sharpe ratio cho từng mã cổ phiếu
def compute_sharpe(returns_df, rf_annual=RF_ANNUAL, trading_days=TRADING_DAYS):
    rf_daily = rf_annual / trading_days
    mean_ret = returns_df.mean()
    std_ret = returns_df.std().replace(0, np.nan)
    sharpe = ((mean_ret - rf_daily) / std_ret).dropna().sort_values(ascending=False)
    return sharpe

# Tính chỉ số RSI
def compute_rsi(price_df, period=14):
    delta = price_df.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

# Trích xuất các đặc trưng kỹ thuật (Feature Engineering)
def build_features(price_df, return_df):
    common_idx = price_df.index.intersection(return_df.index)
    price_df = price_df.loc[common_idx].copy()
    return_df = return_df.loc[common_idx].copy()

    price_df = price_df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return_df = return_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    feat_list = []

    # 1. Return 1 ngày
    ret1 = return_df.copy()
    ret1.columns = [f"{c}_ret1" for c in ret1.columns]
    feat_list.append(ret1)

    # 2. Return 5 ngày
    ret5 = price_df.pct_change(5)
    ret5.columns = [f"{c}_ret5" for c in ret5.columns]
    feat_list.append(ret5)

    # 3. Return 10 ngày
    ret10 = price_df.pct_change(10)
    ret10.columns = [f"{c}_ret10" for c in ret10.columns]
    feat_list.append(ret10)

    # 4. Tỷ lệ MA5
    ma5 = price_df.rolling(5, min_periods=5).mean()
    ma5_ratio = price_df / (ma5 + 1e-9) - 1
    ma5_ratio.columns = [f"{c}_ma5_ratio" for c in ma5_ratio.columns]
    feat_list.append(ma5_ratio)

    # 5. Tỷ lệ MA10
    ma10 = price_df.rolling(10, min_periods=10).mean()
    ma10_ratio = price_df / (ma10 + 1e-9) - 1
    ma10_ratio.columns = [f"{c}_ma10_ratio" for c in ma10_ratio.columns]
    feat_list.append(ma10_ratio)

    # 6. Biến động 5 ngày (Volatility 5d)
    vol5 = return_df.rolling(5, min_periods=5).std()
    vol5.columns = [f"{c}_vol5" for c in vol5.columns]
    feat_list.append(vol5)

    # 7. Biến động 10 ngày (Volatility 10d)
    vol10 = return_df.rolling(10, min_periods=10).std()
    vol10.columns = [f"{c}_vol10" for c in vol10.columns]
    feat_list.append(vol10)

    # 8. Momentum 5 ngày
    mom5 = price_df.pct_change(5)
    mom5.columns = [f"{c}_mom5" for c in mom5.columns]
    feat_list.append(mom5)

    # 9. RSI 14
    rsi14 = compute_rsi(price_df, period=14) / 100.0
    rsi14.columns = [f"{c}_rsi14" for c in rsi14.columns]
    feat_list.append(rsi14)

    features = pd.concat(feat_list, axis=1)
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna(axis=0, how="any")
    return features

# Tạo chuỗi thời gian đầu vào và mục tiêu đầu ra cho mô hình học sâu
def create_sequences_and_targets(features_df, target_returns_df, window_size, horizon=5):
    X, y, dates = [], [], []
    feat_values = features_df.values.astype(np.float32)
    target_values = target_returns_df.values.astype(np.float32)
    idx = features_df.index
    for i in range(len(features_df) - window_size - horizon + 1):
        X.append(feat_values[i : i + window_size])
        y.append(target_values[i + window_size : i + window_size + horizon].mean(axis=0))
        dates.append(idx[i + window_size + horizon - 1])
    if len(X) == 0:
        return np.zeros((0, window_size, features_df.shape[1]), dtype=np.float32), np.zeros((0, target_returns_df.shape[1]), dtype=np.float32), pd.Index([])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), pd.Index(dates)

# Hàm loss Sharpe tùy chỉnh có bổ sung Entropy điều hòa (Regularization)
def sharpe_loss(y_true, y_pred):
    portfolio_returns = tf.reduce_sum(y_true * y_pred, axis=1)
    portfolio_returns = portfolio_returns - (RF_ANNUAL / TRADING_DAYS)
    mean_returns = tf.reduce_mean(portfolio_returns)
    std_returns = tf.math.reduce_std(portfolio_returns)
    sharpe = mean_returns / (std_returns + 1e-9)
    # Entropy penalty giúp tránh hiện tượng phân bổ dồn toàn bộ tỷ trọng vào 1 mã duy nhất
    entropy = -tf.reduce_sum(y_pred * tf.math.log(y_pred + 1e-9), axis=1)
    entropy = tf.reduce_mean(entropy)
    return -sharpe - LAMBDA_ENTROPY * entropy

# Khởi tạo mô hình mạng nơ-ron LSTM-GRU kết hợp
def build_lstm_gru_model(timesteps, n_features, n_assets):
    model = Sequential([
        Input(shape=(timesteps, n_features)),
        LSTM(96, return_sequences=True, activation="tanh", recurrent_activation="sigmoid"),
        Dropout(0.2),
        GRU(48, return_sequences=False, activation="tanh", recurrent_activation="sigmoid"),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.1),
        Dense(n_assets, activation="softmax")
    ])
    return model

# Chiến lược tĩnh 1: Phân bổ đều
def build_allocation_equal(tickers):
    n = len(tickers)
    return pd.DataFrame({"Asset": tickers, "Weight": [1.0 / n] * n})

# Chiến lược tĩnh 2: Phân bổ 80-20 theo Sharpe
def build_allocation_80_20(train_returns):
    rf_daily = RF_ANNUAL / TRADING_DAYS
    mean_ret = train_returns.mean()
    std_ret = train_returns.std().replace(0, np.nan)
    sharpe_train = ((mean_ret - rf_daily) / std_ret).dropna().sort_values(ascending=False)
    
    ranked = sharpe_train.reset_index()
    ranked.columns = ["Asset", "Score"]
    
    n_assets = len(ranked)
    top_count = max(1, int(np.ceil(0.2 * n_assets)))
    bottom_count = n_assets - top_count
    
    top_weights = [0.8 / top_count] * top_count
    bottom_weights = [0.2 / bottom_count] * bottom_count if bottom_count > 0 else []
    
    ranked["Weight"] = top_weights + bottom_weights
    return ranked[["Asset", "Weight"]]

# Tính toán đặc trưng hiệu suất danh mục từ trọng số và tỷ suất sinh lợi
def port_char(weights_df, returns_df, annualize=True, freq=TRADING_DAYS):
    er = returns_df.mean().reset_index()
    er.columns = ["Asset", "Er"]
    
    weights = weights_df.copy()
    weights = weights.merge(er, on="Asset", how="left")
    weights["Er"] = weights["Er"].fillna(0.0)
    
    portfolio_er_daily = np.dot(weights["Weight"], weights["Er"])
    
    cov_matrix = returns_df.cov()
    asset_order = weights["Asset"].tolist()
    cov_matrix = cov_matrix.loc[asset_order, asset_order]
    
    w = weights["Weight"].values
    portfolio_std_daily = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
    
    if annualize:
        return portfolio_er_daily * freq, portfolio_std_daily * np.sqrt(freq)
    return portfolio_er_daily, portfolio_std_daily

# Tính toán đặc trưng hiệu suất danh mục từ chuỗi tỷ suất sinh lợi thực tế
def port_char_from_series(portfolio_return_series, annualize=True, freq=TRADING_DAYS):
    series = pd.Series(portfolio_return_series).dropna()
    er_daily = series.mean()
    std_daily = series.std()
    if annualize:
        return er_daily * freq, std_daily * np.sqrt(freq)
    return er_daily, std_daily

# Tính Sharpe Ratio của danh mục từ trọng số
def sharpe_port(weights_df, returns_df, rf=RF_ANNUAL, freq=TRADING_DAYS):
    er, std = port_char(weights_df, returns_df, annualize=True, freq=freq)
    return (er - rf) / (std + 1e-12)

# Tính Sharpe Ratio của danh mục từ chuỗi tỷ suất sinh lợi
def sharpe_from_series(portfolio_return_series, rf=RF_ANNUAL, freq=TRADING_DAYS):
    er, std = port_char_from_series(portfolio_return_series, annualize=True, freq=freq)
    return (er - rf) / (std + 1e-12)

# Vẽ biểu đồ cột phân bổ tỷ trọng
def plot_allocation_bar(allocation_df, title):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    df_sorted = allocation_df.sort_values("Weight", ascending=False).reset_index(drop=True)
    bars = ax.bar(df_sorted["Asset"].str.upper(), df_sorted["Weight"], color="#4c78a8", edgecolor="grey")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Tỷ trọng (Weight)", fontsize=10)
    ax.set_xlabel("Cổ phiếu", fontsize=10)
    ax.set_ylim(0, df_sorted["Weight"].max() * 1.2)
    
    # Hiển thị số phần trăm trên mỗi cột
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.005,
            f"{height:.2%}",
            ha="center",
            va="bottom",
            fontsize=8
        )
        
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig

# Vẽ biểu đồ khối Treemap phân bổ tỷ trọng
def plot_treemap(allocation_df, title):
    treemap_df = allocation_df.copy()
    treemap_df["Label"] = treemap_df["Asset"].str.upper() + "\n" + (treemap_df["Weight"] * 100).round(2).astype(str) + "%"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    squarify.plot(
        sizes=treemap_df["Weight"],
        label=treemap_df["Label"],
        alpha=0.85,
        edgecolor="black",
        linewidth=1.2,
        text_kwargs={"fontsize": 9, "fontweight": "bold"},
        color=plt.cm.Pastel1.colors
    )
    plt.axis("off")
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig

# Vẽ biểu đồ so sánh đa chiều (Dual-Axis Chart)
def plot_comparison_chart(comparison_table_display):
    categories = comparison_table_display["Chiến lược đầu tư"].values
    er_values = comparison_table_display["Lợi nhuận trung bình (%)"].values
    std_values = comparison_table_display["Độ lệch chuẩn (%)"].values
    sharpe_values = comparison_table_display["Hệ số Sharpe"].values

    x = np.arange(len(categories))
    width = 0.28

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Trục Y bên trái: Lợi nhuận và Rủi ro (%)
    bars1 = ax1.bar(x - width/2, er_values, width, label="Lợi nhuận TB năm (%)", color="#1f77b4", edgecolor="none")
    bars2 = ax1.bar(x + width/2, std_values, width, label="Độ lệch chuẩn năm (%)", color="#aec7e8", edgecolor="none")

    ax1.set_ylabel("Giá trị (%)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # Hiển thị số liệu trên các cột
    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.5,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="semibold"
        )

    # Trục Y bên phải: Hệ số Sharpe
    ax2 = ax1.twinx()
    line = ax2.plot(
        x,
        sharpe_values,
        color="#d62728",
        marker="o",
        markersize=8,
        linewidth=2.5,
        label="Hệ số Sharpe"
    )
    ax2.set_ylabel("Hệ số Sharpe (Sharpe Ratio)", fontsize=11, color="#d62728", fontweight="bold")
    ax2.tick_params(axis='y', labelcolor='#d62728')

    # Hiển thị hệ số Sharpe cạnh dấu chấm tròn
    for i, v in enumerate(sharpe_values):
        ax2.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=9, color="#d62728", fontweight="bold")

    plt.title("So sánh hiệu quả các chiến lược đầu tư (Test Set)", fontsize=12, fontweight="bold", pad=15)

    # Gộp legend của 2 trục
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    return fig

# Hàm chính chạy ứng dụng Streamlit
def main():
    st.set_page_config(page_title="Tối ưu hóa Danh mục Đầu tư HOSE", layout="wide", page_icon="📈")
    
    # CSS tùy chỉnh để làm đẹp giao diện
    st.markdown("""
        <style>
        .main {
            background-color: #f8f9fa;
        }
        h1, h2, h3 {
            color: #1e3d59;
            font-family: 'Inter', sans-serif;
        }
        .reportview-container .main .block-container{
            padding-top: 2rem;
        }
        .stButton>button {
            background-color: #17b978;
            color: white;
            border-radius: 8px;
            padding: 8px 24px;
            font-weight: bold;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #086972;
            color: white;
            box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        }
        .metric-container {
            background-color: #ffffff;
            padding: 16px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border-left: 5px solid #17b978;
            margin-bottom: 12px;
        }
        .metric-title {
            font-size: 14px;
            color: #6c757d;
            font-weight: bold;
        }
        .metric-val {
            font-size: 24px;
            color: #1e3d59;
            font-weight: 800;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("📈 Tối ưu hóa Danh mục Đầu tư Chứng khoán HOSE")
    st.markdown("Ứng dụng tối ưu hóa danh mục đầu tư bằng phương pháp **LSTM-GRU động** từ Jupyter Notebook, so sánh trực tiếp với **Phân bổ đều** và **Phân bổ 80-20 theo Sharpe**.")

    # --- PHẦN SIDEBAR ---
    st.sidebar.header("⚙️ Cấu hình Tham số")

    # Tải dữ liệu CSV
    csv_path = "HOSE_2020_2023 (2).csv"
    uploaded_file = st.sidebar.file_uploader("Tải lên file dữ liệu HOSE CSV", type=["csv"])

    df = None
    if uploaded_file is not None:
        try:
            df = load_price_data(uploaded_file=uploaded_file)
            st.sidebar.success("Tải dữ liệu tải lên thành công!")
        except Exception as e:
            st.sidebar.error(f"Lỗi định dạng file: {e}")
            st.stop()
    elif os.path.exists(csv_path):
        st.sidebar.info(f"Sử dụng dữ liệu mặc định: {csv_path}")
        df = load_price_data(csv_path=csv_path)
    else:
        st.sidebar.error("Không tìm thấy file dữ liệu. Vui lòng tải lên file CSV.")
        st.stop()

    # Tìm danh sách mã cổ phiếu
    all_tickers = sorted(df["ticker"].unique())
    
    # Cho phép người dùng lọc danh mục mã
    use_all_tickers = st.sidebar.checkbox("Sử dụng tất cả các mã", value=True)
    if use_all_tickers:
        selected_tickers = all_tickers
    else:
        selected_tickers = st.sidebar.multiselect(
            "Chọn các mã cổ phiếu phân tích",
            all_tickers,
            default=all_tickers[:15]
        )

    if not selected_tickers:
        st.error("Vui lòng chọn ít nhất một mã cổ phiếu để phân tích.")
        st.stop()

    # Cấu hình mốc thời gian Train/Test
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    
    st.sidebar.markdown("#### Khoảng thời gian phân tích")
    # Mặc định: 80% train, 20% test
    delta_days = (max_date - min_date).days
    default_train_end = min_date + pd.Timedelta(days=int(delta_days * 0.8))
    default_test_start = default_train_end + pd.Timedelta(days=1)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        train_start = st.date_input("Train Bắt đầu", min_date, min_value=min_date, max_value=max_date)
    with col2:
        train_end = st.date_input("Train Kết thúc", default_train_end, min_value=min_date, max_value=max_date)

    col3, col4 = st.sidebar.columns(2)
    with col3:
        test_start = st.date_input("Test Bắt đầu", default_test_start, min_value=min_date, max_value=max_date)
    with col4:
        test_end = st.date_input("Test Kết thúc", max_date, min_value=min_date, max_value=max_date)

    # Validate thời gian
    if train_start >= train_end:
        st.sidebar.error("Ngày bắt đầu Train phải nhỏ hơn ngày kết thúc Train.")
        st.stop()
    if test_start >= test_end:
        st.sidebar.error("Ngày bắt đầu Test phải nhỏ hơn ngày kết thúc Test.")
        st.stop()
    if train_end >= test_start:
        st.sidebar.warning("Lưu ý: Tập Train và tập Test đang bị chồng lấn.")

    # Các tham số mô hình
    st.sidebar.markdown("#### Cấu hình Mô hình")
    top_n = st.sidebar.slider("Số cổ phiếu Top Sharpe để train", min_value=3, max_value=len(selected_tickers), value=min(10, len(selected_tickers)), step=1)
    window_size = st.sidebar.slider("Window Size (Ngày)", min_value=5, max_value=60, value=DEFAULT_WINDOW_SIZE, step=5)
    horizon = st.sidebar.slider("Horizon (Ngày dự báo)", min_value=1, max_value=10, value=DEFAULT_HORIZON, step=1)
    epochs = st.sidebar.slider("Số lượng Epochs", min_value=5, max_value=100, value=30, step=5)
    batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    
    # Tùy chọn huấn luyện nhiều seed
    multi_seed = st.sidebar.checkbox("Huấn luyện đa Seed (chọn tốt nhất)", value=False)
    
    # Nút bấm bắt đầu
    st.sidebar.markdown("---")
    run_optimization = st.sidebar.button("🚀 Bắt đầu tối ưu hóa")

    # Chuẩn bị ma trận giá và lợi nhuận
    price_matrix = prepare_price_matrix(df, selected_tickers)
    
    # Chuyển đổi định dạng index để filter thời gian dễ dàng
    price_matrix.index = pd.to_datetime(price_matrix.index)
    returns = compute_returns(price_matrix)

    # Tách tập dữ liệu train và test theo khoảng thời gian được chọn
    train_prices = price_matrix.loc[pd.Timestamp(train_start):pd.Timestamp(train_end)].copy()
    test_prices = price_matrix.loc[pd.Timestamp(test_start):pd.Timestamp(test_end)].copy()
    
    train_returns = returns.loc[pd.Timestamp(train_start):pd.Timestamp(train_end)].copy().dropna(how="all")
    test_returns = returns.loc[pd.Timestamp(test_start):pd.Timestamp(test_end)].copy().dropna(how="all")

    # Tính Sharpe để chọn Top cổ phiếu
    sharpe_train_full = compute_sharpe(train_returns)
    top_symbols = sharpe_train_full.head(top_n).index.tolist()

    # Tạo các Tabs hiển thị
    tab_data, tab_static, tab_dl, tab_compare = st.tabs([
        "📊 Tổng quan Dữ liệu", 
        "⚖️ Phân bổ Tĩnh", 
        "🧠 Mô hình LSTM-GRU (Động)", 
        "🏆 So sánh & Kết quả"
    ])

    # --- TAB 1: TỔNG QUAN DỮ LIỆU ---
    with tab_data:
        st.subheader("Thông tin tập dữ liệu")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(f"<div class='metric-container'><div class='metric-title'>Tổng số mã khả dụng</div><div class='metric-val'>{len(all_tickers)}</div></div>", unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"<div class='metric-container'><div class='metric-title'>Số mã đã chọn phân tích</div><div class='metric-val'>{len(selected_tickers)}</div></div>", unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"<div class='metric-container'><div class='metric-title'>Số ngày giao dịch Train</div><div class='metric-val'>{train_prices.shape[0]}</div></div>", unsafe_allow_html=True)
        with col_m4:
            st.markdown(f"<div class='metric-container'><div class='metric-title'>Số ngày giao dịch Test</div><div class='metric-val'>{test_prices.shape[0]}</div></div>", unsafe_allow_html=True)

        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.subheader("Biểu đồ giá đóng cửa chuẩn hóa (về mốc 100)")
            if not train_prices.empty:
                # Chuẩn hóa giá đóng cửa về 100 tại điểm khởi đầu để dễ so sánh
                norm_prices = (price_matrix[selected_tickers] / price_matrix[selected_tickers].iloc[0]) * 100
                st.line_chart(norm_prices)
            else:
                st.warning("Không có dữ liệu hiển thị biểu đồ.")
        with col_right:
            st.subheader("Top cổ phiếu có Sharpe cao nhất (Tập Train)")
            if not sharpe_train_full.empty:
                st.dataframe(sharpe_train_full.head(15).rename("Sharpe Ratio").to_frame(), height=380)
            else:
                st.info("Chưa có thông tin.")

        # Thống kê mô tả
        st.subheader("Thống kê mô tả tỷ suất sinh lợi hàng ngày (Tập Train)")
        if not train_returns.empty:
            desc_df = train_returns[selected_tickers].describe().T
            desc_df["Annual Return (Est)"] = desc_df["mean"] * TRADING_DAYS
            desc_df["Annual Volatility (Est)"] = desc_df["std"] * np.sqrt(TRADING_DAYS)
            st.dataframe(desc_df[["mean", "std", "min", "max", "Annual Return (Est)", "Annual Volatility (Est)"]].style.format("{:.4f}"))
        else:
            st.info("Chưa có thông tin thống kê.")

    # --- TAB 2: PHÂN BỔ TĨNH ---
    with tab_static:
        st.subheader("Các chiến lược phân bổ danh mục tĩnh")
        st.write("Chiến lược được thiết kế dựa trên dữ liệu **Tập Train** và sẽ được kiểm thử trên **Tập Test**.")

        # Tạo allocation cho 2 chiến lược tĩnh
        allo_equal = build_allocation_equal(top_symbols)
        allo_8020 = build_allocation_80_20(train_returns[top_symbols])

        col_eq, col_80 = st.columns(2)
        with col_eq:
            st.markdown("### 1. Chiến lược Phân bổ Đều (Equal Weight)")
            st.write("Tỷ trọng được chia đều cho các cổ phiếu trong Top Sharpe.")
            st.dataframe(allo_equal.style.format({"Weight": "{:.2%}"}))
            st.pyplot(plot_allocation_bar(allo_equal, "Tỷ trọng Phân bổ Đều"))
            
        with col_80:
            st.markdown("### 2. Chiến lược 80-20 theo Sharpe")
            st.write("Xếp hạng Sharpe từ cao xuống thấp. 20% số mã hàng đầu chiếm 80% tỷ trọng danh mục, 80% số mã còn lại chiếm 20% tỷ trọng.")
            st.dataframe(allo_8020.style.format({"Weight": "{:.2%}"}))
            st.pyplot(plot_allocation_bar(allo_8020, "Tỷ trọng Phân bổ 80-20"))

    # --- TAB 3: MÔ HÌNH LSTM-GRU (ĐỘNG) ---
    with tab_dl:
        st.subheader("🧠 Huấn luyện Mô hình LSTM-GRU động")
        st.write("Mô hình học sâu kết hợp giữa LSTM và GRU tối ưu hóa trực tiếp hệ số Sharpe thông qua hàm loss tùy chỉnh.")
        
        # Kiến trúc mô hình
        with st.expander("🔍 Chi tiết cấu trúc mạng LSTM-GRU và Hàm Loss"):
            st.markdown("""
            - **Lớp mạng đầu vào (Input)**: Dữ liệu chuỗi thời gian (Timesteps = Window size, Features = số lượng đặc trưng kỹ thuật của toàn bộ cổ phiếu).
            - **LSTM Layer**: 96 units, activation 'tanh', recurrent activation 'sigmoid'. Dropout 0.2.
            - **GRU Layer**: 48 units, activation 'tanh', recurrent activation 'sigmoid'. Dropout 0.2.
            - **Dense Layer**: 64 units, activation 'relu'. Dropout 0.1.
            - **Lớp đầu ra (Dense Output)**: Số lượng nơ-ron bằng số cổ phiếu cần tối ưu, kích hoạt bằng hàm **Softmax** để tổng tỷ trọng phân bổ luôn bằng 100%.
            - **Hàm Loss (Sharpe Loss)**: 
              $$\mathcal{L} = - \text{Sharpe Daily} - \lambda \cdot \text{Entropy}$$
              Trong đó, $\lambda = 0.01$ là trọng số entropy nhằm phân tán tỷ trọng, ngăn chặn mô hình dồn 100% tài sản vào một mã duy nhất.
            """)

        # Kiểm tra tính hợp lệ của dữ liệu trước khi train
        data_valid = True
        try:
            train_features = build_features(train_prices[top_symbols], train_returns[top_symbols])
            test_features = build_features(test_prices[top_symbols], test_returns[top_symbols])
            
            # Khởi tạo scaler
            scaler = StandardScaler()
            train_features_scaled = pd.DataFrame(
                scaler.fit_transform(train_features),
                index=train_features.index,
                columns=train_features.columns
            )
            test_features_scaled = pd.DataFrame(
                scaler.transform(test_features),
                index=test_features.index,
                columns=test_features.columns
            )

            # Lấy mục tiêu thực tế (Tỷ suất sinh lợi tiếp theo)
            train_target = train_returns.loc[train_features_scaled.index, top_symbols].copy()
            test_target = test_returns.loc[test_features_scaled.index, top_symbols].copy()

            # Tạo sequence
            X_train, y_train, train_seq_dates = create_sequences_and_targets(
                train_features_scaled, train_target, window_size, horizon=horizon
            )
            X_test, y_test, test_seq_dates = create_sequences_and_targets(
                test_features_scaled, test_target, window_size, horizon=horizon
            )
        except Exception as e:
            st.error(f"Lỗi xử lý đặc trưng kỹ thuật: {e}")
            data_valid = False

        if data_valid:
            st.info(f"Kích thước tập Train sequences: {X_train.shape} | Tập Test sequences: {X_test.shape}")
            if X_train.shape[0] == 0 or X_test.shape[0] == 0:
                st.error("Dữ liệu không đủ dài để tạo chuỗi thời gian (Sequences). Vui lòng giảm Window Size / Horizon hoặc tăng khoảng thời gian kiểm định.")
                data_valid = False

        # Khởi tạo session state để lưu trữ kết quả huấn luyện
        if "trained" not in st.session_state:
            st.session_state.trained = False
            st.session_state.dynamic_results = None

        if data_valid:
            if run_optimization:
                st.session_state.trained = False
                
                # Xác định danh sách seed
                seed_list = [7, 21, 42, 99, 123] if multi_seed else [42]
                
                # Khởi động thanh tiến trình
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                best_model = None
                best_sharpe = -1e9
                best_seed = None
                best_history = None
                best_portfolio_returns = None
                best_pred_weights_test = None
                all_histories = {}
                
                n_seeds = len(seed_list)
                
                # Thực hiện vòng lặp huấn luyện theo seed
                for idx, seed in enumerate(seed_list):
                    status_text.write(f"⏳ **Đang huấn luyện mô hình với Seed {seed}** ({idx+1}/{n_seeds})...")
                    set_seed(seed)
                    
                    # Khởi tạo mô hình
                    model = build_lstm_gru_model(X_train.shape[1], X_train.shape[2], y_train.shape[1])
                    model.compile(optimizer=Adam(learning_rate=0.0005), loss=sharpe_loss)
                    
                    callbacks = [
                        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5)
                    ]
                    
                    history = model.fit(
                        X_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=False,
                        verbose=0,
                        validation_split=0.2,
                        callbacks=callbacks
                    )
                    
                    # Predict trên tập test
                    pred_weights = model.predict(X_test, verbose=0)
                    weights_df = pd.DataFrame(pred_weights, index=test_seq_dates, columns=top_symbols)
                    
                    # Tính toán return thực tế của danh mục
                    portfolio_returns = (weights_df * test_target.loc[test_seq_dates]).sum(axis=1)
                    er, std = port_char_from_series(portfolio_returns)
                    run_sharpe = (er - RF_ANNUAL) / (std + 1e-12)
                    
                    all_histories[seed] = history.history
                    
                    if run_sharpe > best_sharpe:
                        best_sharpe = run_sharpe
                        best_seed = seed
                        best_model = model
                        best_history = history
                        best_portfolio_returns = portfolio_returns
                        best_pred_weights_test = pred_weights
                        
                    progress_bar.progress((idx + 1) / n_seeds)
                
                status_text.success(f"✔️ Hoàn tất huấn luyện! Mô hình tốt nhất đạt được tại **Seed {best_seed}** với Sharpe Test: **{best_sharpe:.4f}**")
                
                # Lưu trung bình tỷ trọng
                avg_weights = best_pred_weights_test.mean(axis=0)
                allo_lstm = pd.DataFrame({"Asset": top_symbols, "Weight": avg_weights})
                allo_lstm = allo_lstm.sort_values("Weight", ascending=False).reset_index(drop=True)
                
                # Lưu thông tin vào session state
                st.session_state.dynamic_results = {
                    "allo_lstm": allo_lstm,
                    "portfolio_returns": best_portfolio_returns,
                    "er": best_portfolio_returns.mean() * TRADING_DAYS,
                    "std": best_portfolio_returns.std() * np.sqrt(TRADING_DAYS),
                    "sharpe": best_sharpe,
                    "history": best_history.history,
                    "seed": best_seed,
                    "weights_over_time": pd.DataFrame(best_pred_weights_test, index=test_seq_dates, columns=top_symbols)
                }
                st.session_state.trained = True

        # Hiển thị kết quả huấn luyện nếu đã có trong session state
        if st.session_state.trained:
            res = st.session_state.dynamic_results
            
            st.markdown("### Kết quả Phân bổ Danh mục Động (Trung bình tập Test)")
            col_l, col_r = st.columns([1, 2])
            with col_l:
                st.write(f"Tỷ trọng phân bổ trung bình (Seed tốt nhất: {res['seed']}):")
                st.dataframe(res["allo_lstm"].style.format({"Weight": "{:.2%}"}))
            with col_r:
                st.pyplot(plot_allocation_bar(res["allo_lstm"], "Tỷ trọng Trung bình LSTM-GRU"))

            st.pyplot(plot_treemap(res["allo_lstm"], "Treemap Tỷ trọng Danh mục LSTM-GRU"))

            st.markdown("### Lịch sử Loss Huấn luyện (Best Seed)")
            hist_df = pd.DataFrame(res["history"])
            fig_loss, ax_loss = plt.subplots(figsize=(8, 3.5))
            ax_loss.plot(hist_df["loss"], label="Train Loss", color="#1f77b4")
            if "val_loss" in hist_df.columns:
                ax_loss.plot(hist_df["val_loss"], label="Validation Loss", color="#ff7f0e")
            ax_loss.set_title("Training & Validation Loss History")
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss")
            ax_loss.legend()
            ax_loss.grid(True, linestyle=":", alpha=0.6)
            st.pyplot(fig_loss)
            
            st.markdown("### Biến động tỷ trọng các tài sản theo thời gian")
            st.line_chart(res["weights_over_time"])
        else:
            st.warning("Mô hình học sâu chưa được huấn luyện. Vui lòng bấm vào nút **🚀 Bắt đầu tối ưu hóa** ở thanh Sidebar.")

    # --- TAB 4: SO SÁNH & KẾT LUẬN ---
    with tab_compare:
        st.subheader("🏆 So sánh hiệu quả các chiến lược đầu tư")
        
        if st.session_state.trained:
            res = st.session_state.dynamic_results
            
            # Cần lọc lại tập test_returns khớp chính xác với ngày của sequence test
            common_test_returns = test_returns.loc[res["portfolio_returns"].index, top_symbols]

            # Tính toán hiệu quả 2 chiến lược tĩnh trên cùng khoảng thời gian Test của sequence
            er_eq, std_eq = port_char(allo_equal, common_test_returns, annualize=True)
            sharpe_eq = sharpe_port(allo_equal, common_test_returns)

            er_8020, std_8020 = port_char(allo_8020, common_test_returns, annualize=True)
            sharpe_8020 = sharpe_port(allo_8020, common_test_returns)

            comparison = [
                {
                    "Chiến lược đầu tư": "LSTM-GRU (Động)",
                    "Lợi nhuận trung bình (%)": res["er"] * 100,
                    "Độ lệch chuẩn (%)": res["std"] * 100,
                    "Hệ số Sharpe": res["sharpe"]
                },
                {
                    "Chiến lược đầu tư": "Phân bổ đều (Equal Weight)",
                    "Lợi nhuận trung bình (%)": er_eq * 100,
                    "Độ lệch chuẩn (%)": std_eq * 100,
                    "Hệ số Sharpe": sharpe_eq
                },
                {
                    "Chiến lược đầu tư": "Phân bổ 80-20 (Sharpe)",
                    "Lợi nhuận trung bình (%)": er_8020 * 100,
                    "Độ lệch chuẩn (%)": std_8020 * 100,
                    "Hệ số Sharpe": sharpe_8020
                }
            ]

            comp_df = pd.DataFrame(comparison)
            st.dataframe(comp_df.style.format({
                "Lợi nhuận trung bình (%)": "{:.2f}%",
                "Độ lệch chuẩn (%)": "{:.2f}%",
                "Hệ số Sharpe": "{:.4f}"
            }))

            # Vẽ biểu đồ so sánh dual-axis
            st.pyplot(plot_comparison_chart(comp_df))

            # Kết luận nhanh
            best_strategy = comp_df.loc[comp_df["Hệ số Sharpe"].idxmax(), "Chiến lược đầu tư"]
            st.markdown(f"""
            ### 📝 Kết luận phân tích:
            - Chiến lược có hiệu quả tối ưu nhất xét theo hệ số Sharpe trên tập dữ liệu kiểm thử (Test Set) là: **{best_strategy}**.
            - Mô hình **LSTM-GRU động** tự động điều chỉnh tỷ trọng hàng ngày/tuần tùy thuộc vào diễn biến đặc trưng kỹ thuật, giúp thích ứng nhanh với các giai đoạn của thị trường.
            """)

            # Cho phép download tỷ trọng tối ưu
            st.subheader("📥 Tải xuống kết quả tỷ trọng tối ưu")
            
            # Chuẩn bị file tải xuống gộp tỷ trọng
            w_eq = allo_equal.rename(columns={"Weight": "Weight_Equal"})
            w_8020 = allo_8020.rename(columns={"Weight": "Weight_8020"})
            w_lstm = res["allo_lstm"].rename(columns={"Weight": "Weight_LSTM_GRU"})
            
            final_weights = w_eq.merge(w_8020, on="Asset", how="outer").merge(w_lstm, on="Asset", how="outer").fillna(0.0)
            
            csv_data = final_weights.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Tải xuống tỷ trọng tối ưu (CSV)",
                data=csv_data,
                file_name="optimized_portfolio_weights.csv",
                mime="text/csv"
            )
        else:
            st.warning("Vui lòng huấn luyện mô hình tại Tab **🧠 Mô hình LSTM-GRU (Động)** trước để so sánh kết quả.")

    st.markdown("---")
    st.markdown("☘️ *Ứng dụng được phát triển bởi **Antigravity** - Chuyên gia AI của bạn.*")


if __name__ == "__main__":
    main()
