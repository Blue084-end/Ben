import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from supabase import create_client
import uuid

# 🔐 Kết nối Supabase bằng st.secrets
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# 🎨 Cấu hình giao diện
st.set_page_config(page_title="Baccarat Predictor Pro", layout="wide")
st.title("🎲 Baccarat Predictor Pro")

# 📧 Nhập email người dùng
user_email = st.text_input("📧 Nhập email để bắt đầu:", key="email")

# Tabs giao diện
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Dự đoán", "📊 Phân tích", "📜 Lịch sử", "🛠 Quản lý dữ liệu"])

# Session state
if "data" not in st.session_state:
    st.session_state["data"] = []
if "model" not in st.session_state:
    st.session_state["model"] = None
if "replay" not in st.session_state:
    st.session_state["replay"] = []

# 🔮 Tab 1: Dự đoán
with tab1:
    st.subheader("📥 Nhập kết quả mới")
    result = st.radio("Chọn kết quả ván vừa rồi:", ["Player", "Banker", "Tie"])
    if st.button("➕ Thêm kết quả"):
        st.session_state["data"].append(result)
        st.session_state["replay"].append(result)
        if user_email:
            supabase.table("baccarat_results").insert({
                "email": user_email,
                "result": result,
                "timestamp": pd.Timestamp.now().isoformat()
            }).execute()

    st.subheader("📋 Dữ liệu đã nhập")
    df = pd.DataFrame(st.session_state["data"], columns=["Result"])
    st.dataframe(df, use_container_width=True)

    def encode_result(r):
        return {"Player": 0, "Banker": 1, "Tie": 2}[r]

    if len(st.session_state["data"]) >= 5:
        encoded = [encode_result(r) for r in st.session_state["data"]]
        X, y = [], []
        for i in range(len(encoded) - 3):
            X.append(encoded[i:i+3])
            y.append(encoded[i+3])
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        st.session_state["model"] = model
        latest = encoded[-3:]
        prediction = model.predict([latest])[0]
        pred_label = ["Player", "Banker", "Tie"][prediction]
        st.success(f"🔮 Dự đoán tiếp theo: **{pred_label}**")

        # Lưu phiên chơi
        stats = {
            "total_games": len(st.session_state["data"]),
            "player_count": st.session_state["data"].count("Player"),
            "banker_count": st.session_state["data"].count("Banker"),
            "tie_count": st.session_state["data"].count("Tie")
        }
        session_id = str(uuid.uuid4())
        supabase.table("baccarat_sessions").insert({
            "email": user_email,
            "session_id": session_id,
            "model_info": {"n_estimators": 100, "random_state": 42},
            "stats": stats,
            "timestamp": pd.Timestamp.now().isoformat()
        }).execute()
    else:
        st.info("Cần ít nhất 5 kết quả để bắt đầu dự đoán.")

# 📊 Tab 2: Phân tích
with tab2:
    st.subheader("📈 Biểu đồ tần suất kết quả")
    if not df.empty:
        fig, ax = plt.subplots()
        sns.countplot(x="Result", data=df, ax=ax, palette="Set2")
        ax.set_title("Tần suất Player / Banker / Tie")
        st.pyplot(fig)
    else:
        st.info("Chưa có dữ liệu để hiển thị biểu đồ.")

    st.subheader("🚨 Cảnh báo chuỗi lặp")
    def detect_streak(data, threshold=4):
        streaks = []
        count = 1
        for i in range(1, len(data)):
            if data[i] == data[i-1]:
                count += 1
                if count >= threshold:
                    streaks.append((data[i], count))
            else:
                count = 1
        return streaks

    streaks = detect_streak(st.session_state["data"])
    if streaks:
        for s in streaks:
            st.warning(f"⚠️ Chuỗi {s[0]} lặp lại {s[1]} lần liên tiếp!")
    else:
        st.success("✅ Không có chuỗi lặp bất thường.")

    st.subheader("⏮️ Replay lịch sử")
    if st.session_state["replay"]:
        replay_df = pd.DataFrame(st.session_state["replay"], columns=["Lịch sử"])
        st.dataframe(replay_df, use_container_width=True)
    else:
        st.info("Chưa có lịch sử để hiển thị.")

# 📜 Tab 3: Lịch sử phiên chơi
with tab3:
    st.subheader("📜 Lịch sử phiên chơi")
    start_date = st.date_input("📅 Từ ngày", value=pd.Timestamp.now().date() - pd.Timedelta(days=7))
    end_date = st.date_input("📅 Đến ngày", value=pd.Timestamp.now().date())

    def get_sessions(email):
        response = supabase.table("baccarat_sessions").select("*").eq("email", email).order("timestamp", desc=True).execute()
        return response.data

    if user_email:
        sessions = get_sessions(user_email)
        filtered = [s for s in sessions if start_date <= pd.to_datetime(s["timestamp"]).date() <= end_date]
        for s in filtered:
            with st.expander(f"🧾 Phiên {s['session_id']} - {s['timestamp']}"):
                st.json(s["model_info"])
                st.json(s["stats"])
    else:
        st.info("Vui lòng nhập email để xem lịch sử.")

# 🛠 Tab 4: Quản lý dữ liệu
with tab4:
    st.subheader("🛠 Quản lý dữ liệu")
    def get_user_data(email):
        response = supabase.table("baccarat_results").select("*").eq("email", email).order("timestamp", desc=True).execute()
        return response.data

    def update_result(record_id, new_result):
        return supabase.table("baccarat_results").update({
            "result": new_result,
            "timestamp": pd.Timestamp.now().isoformat()
        }).eq("id", record_id).execute()

    def delete_result(record_id):
        return supabase.table("baccarat_results").delete().eq("id", record_id).execute()

    if user_email:
        start = st.date_input("📅 Từ ngày", value=pd.Timestamp.now().date() - pd.Timedelta(days=7), key="filter_start")
        end = st.date_input("📅 Đến ngày", value=pd.Timestamp.now().date(), key="filter_end")

        user_data = get_user_data(user_email)
        df = pd.DataFrame(user_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        filtered_df = df[(df["timestamp"].dt.date >= start) & (df["timestamp"].dt.date <= end)]

        st.dataframe(filtered_df[["id", "result", "timestamp"]], use_container_width=True)

        selected_id = st.selectbox("🔍 Chọn ID để chỉnh sửa hoặc xóa:", filtered_df["id"])
        selected_row = filtered_df[filtered_df["id"] == selected_id].iloc[0]
        new_result = st.selectbox("✏️ Chỉnh sửa kết quả:", ["Player", "Banker", "Tie"], index=["Player", "Banker", "Tie"].index(selected_row["result"]))

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Cập nhật kết quả"):
                update_result(selected_id, new_result)
                st.success("Đã cập nhật thành công!")
        with col2:
            if st.button("🗑️ Xóa bản ghi"):
                confirm = st.radio("❓ Bạn có chắc muốn xóa?", ["Không", "Có"], index=0)
                if confirm == "Có":
                    delete_result(selected_id)
                    st.warning("Đã xóa bản ghi!")
    else:
        st.info("Vui lòng nhập email để quản lý dữ liệu.")
