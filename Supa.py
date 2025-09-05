import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim

# Cấu hình trang
st.set_page_config(page_title="Baccarat AI", layout="wide")

# Khởi tạo session state
for key in ["data", "new_result", "trained_model", "accuracy"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "data" else None if key == "trained_model" else ""

# Sidebar chọn mô hình
st.sidebar.title("🔧 Tuỳ chọn mô hình dự đoán")
selected_model = st.sidebar.radio("Chọn mô hình", ["Phân tích & Dự đoán", "Tổng quan", "Markov Chain", "RNN (LSTM)", "GRU", "RNN (PyTorch)"])
show_markov = st.sidebar.checkbox("🔁 Bật Markov Chain", value=(selected_model == "Markov Chain"))
show_lstm = st.sidebar.checkbox("🔮 Bật RNN (LSTM)", value=(selected_model == "RNN (LSTM)"))
show_gru = st.sidebar.checkbox("⚡ Bật GRU", value=(selected_model == "GRU"))
show_torch = st.sidebar.checkbox("🔥 Bật RNN (PyTorch)", value=(selected_model == "RNN (PyTorch)"))
epochs = st.sidebar.slider("Số epoch huấn luyện", min_value=5, max_value=50, value=15)

# Giao diện chính
if selected_model == "Phân tích & Dự đoán":
    st.markdown("<h4 style='margin-bottom:10px;'>🎲 Phân tích & Dự đoán Baccarat</h4>", unsafe_allow_html=True)

    # Nhập dữ liệu viết tắt
    input_result = st.text_input("Nhập kết quả (P, B, T):", key="new_result")
    mapping = {"P": "Player", "B": "Banker", "T": "Tie"}

    if input_result:
        result = input_result.strip().upper()
        if result in mapping:
            st.session_state["data"].append(mapping[result])
            st.session_state["new_result"] = ""
        else:
            st.error("❌ Ký tự không hợp lệ. Chỉ dùng P, B, T.")

    # Hiển thị tiêu đề + kết quả mới thêm cùng hàng
    if st.session_state["data"]:
        last_added = st.session_state["data"][-1]
        st.markdown(f"""
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <h5 style='margin:0;'>📋 Dữ liệu hiện tại</h5>
                <span style='color:green; font-weight:bold;'>✅ Đã thêm: {last_added}</span>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Chưa có dữ liệu.")

    # Menu ẩn: xóa & hoàn tác
    with st.expander("⚙️ Tuỳ chọn dữ liệu"):
        if st.button("🗑️ Xóa toàn bộ dữ liệu"):
            st.session_state["data"] = []
            st.success("✅ Đã xóa toàn bộ dữ liệu.")
        if st.session_state["data"]:
            last_result = st.session_state["data"][-1]
            if st.button(f"↩️ Hoàn tác kết quả cuối: {last_result}"):
                st.session_state["data"].pop()
                st.success(f"✅ Đã hoàn tác: {last_result}")

    # Hiển thị bảng dữ liệu dạng Excel
    if st.session_state["data"]:
        df_data = st.session_state["data"]
        max_cols = 6
        rows = int(np.ceil(len(df_data) / max_cols))
        matrix = [["" for _ in range(max_cols)] for _ in range(rows)]

        for idx, val in enumerate(df_data):
            r, c = divmod(idx, max_cols)
            matrix[r][c] = val

        df_matrix = pd.DataFrame(matrix)

        def highlight(val):
            styles = {
                "Player": "color:green; font-weight:bold",
                "Banker": "color:red; font-weight:bold",
                "Tie": "color:orange; font-weight:bold"
            }
            return styles.get(val, "")

        st.dataframe(df_matrix.style.applymap(highlight), use_container_width=True)



# Tổng quan hệ thống
if selected_model == "Tổng quan":
    st.subheader("📊 Tổng quan hệ thống")
    st.markdown("""
    - ✅ Nhập kết quả Baccarat theo thời gian thực  
    - 🔁 Phân tích Markov Chain để hiểu xu hướng chuyển tiếp  
    - 🔮 Dự đoán kết quả tiếp theo bằng RNN (LSTM, GRU, PyTorch)  
    - 📈 Hiển thị xác suất dự đoán  
    - 🧭 Tuỳ chọn bật/tắt từng mô hình trong sidebar
    """)

# Hàm xây dựng Markov Chain
def build_markov_chain(data):
    states = ["Player", "Banker", "Tie"]
    matrix = pd.DataFrame(0, index=states, columns=states)
    for i in range(len(data) - 1):
        matrix.loc[data[i], data[i + 1]] += 1
    prob_matrix = matrix.div(matrix.sum(axis=1), axis=0).fillna(0)
    return matrix, prob_matrix

# Hiển thị Markov Chain nếu được bật
if show_markov:
    st.subheader("🔁 Phân tích Markov Chain")
    if len(st.session_state["data"]) >= 2:
        count_matrix, prob_matrix = build_markov_chain(st.session_state["data"])
        st.dataframe(prob_matrix.style.format("{:.2f}"))
        fig, ax = plt.subplots()
        sns.heatmap(prob_matrix, annot=True, cmap="Blues", fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.info("⚠️ Cần ít nhất 2 kết quả để phân tích.")






# Mã hóa dữ liệu
def encode_sequence(data):
    mapping = {"Player": 0, "Banker": 1, "Tie": 2}
    return [mapping[d] for d in data if d in mapping]

# Tạo chuỗi dữ liệu đầu vào
def create_sequences(encoded, seq_length=5):
    X, y = [], []
    for i in range(len(encoded) - seq_length):
        X.append(encoded[i:i+seq_length])
        y.append(encoded[i+seq_length])
    return np.array(X).reshape(-1, seq_length, 1), np.array(y)

# Xây dựng mô hình RNN bằng TensorFlow
def build_model(model_type="LSTM", seq_length=5, num_classes=3):
    RNNLayer = tf.keras.layers.LSTM if model_type == "LSTM" else tf.keras.layers.GRU
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_length, 1)),
        RNNLayer(64),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Huấn luyện mô hình TensorFlow
def train_tf_model(data, model_type="LSTM", seq_length=5, epochs=30):
    encoded = encode_sequence(data)
    X, y = create_sequences(encoded, seq_length)
    model = build_model(model_type, seq_length)
    history = model.fit(X, y, epochs=epochs, verbose=0)
    return model, history

# Dự đoán bằng mô hình TensorFlow
def predict_tf(model, data, seq_length=5):
    encoded = encode_sequence(data)
    if len(encoded) < seq_length:
        return "Không đủ dữ liệu", [0, 0, 0]
    input_seq = np.array(encoded[-seq_length:]).reshape(1, seq_length, 1)
    pred = model.predict(input_seq)[0]
    mapping = {0: "Player", 1: "Banker", 2: "Tie"}
    return mapping[np.argmax(pred)], pred

# Giao diện dự đoán bằng LSTM hoặc GRU
if show_lstm or show_gru:
    model_type = "LSTM" if show_lstm else "GRU"
    st.subheader(f"🔮 Dự đoán Baccarat bằng {model_type}")
    if len(st.session_state["data"]) >= 10:
        if st.button("🔮 Dự đoán tiếp theo"):
            if st.session_state["trained_model"] is None:
                model, history = train_tf_model(st.session_state["data"], model_type, epochs=epochs)
                st.session_state["trained_model"] = model
                accuracy = history.history["accuracy"][-1]
                st.session_state["accuracy"] = accuracy
                st.write(f"✅ Độ chính xác mô hình {model_type}: {round(accuracy * 100, 2)}%")
            else:
                st.write("✅ Mô hình đã được huấn luyện trước đó.")
                accuracy = st.session_state.get("accuracy", None)
                if accuracy is not None:
                    st.write(f"✅ Độ chính xác mô hình {model_type}: {round(accuracy * 100, 2)}%")
                else:
                    st.info("ℹ️ Chưa có thông tin độ chính xác.")
    else:
        st.warning("⚠️ Cần ít nhất 10 kết quả để huấn luyện mô hình.")






# Mô hình RNN sử dụng PyTorch
class RNNModelTorch(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=3):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)

# Huấn luyện mô hình PyTorch
def train_torch_model(data, seq_length=5, epochs=30):
    encoded = encode_sequence(data)
    X, y = create_sequences(encoded, seq_length)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    model = RNNModelTorch()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
    return model

# Dự đoán bằng mô hình PyTorch
def predict_torch(model, data, seq_length=5):
    encoded = encode_sequence(data)
    if len(encoded) < seq_length:
        return "Không đủ dữ liệu", [0, 0, 0]
    input_seq = torch.tensor(encoded[-seq_length:], dtype=torch.float32).reshape(1, seq_length, 1)
    model.eval()
    with torch.no_grad():
        output = model(input_seq)
        probs = torch.softmax(output, dim=1).numpy()[0]
    mapping = {0: "Player", 1: "Banker", 2: "Tie"}
    return mapping[np.argmax(probs)], probs

# Giao diện dự đoán bằng PyTorch
if show_torch:
    st.subheader("🔥 Dự đoán Baccarat bằng RNN (PyTorch)")
    if len(st.session_state["data"]) >= 10:
        if st.button("🔥 Dự đoán tiếp theo"):
            model = train_torch_model(st.session_state["data"], epochs=epochs)
            prediction, probs = predict_torch(model, st.session_state["data"])
            st.success(f"✅ Dự đoán: {prediction}")
            st.write(f"🔢 Xác suất: Player={round(probs[0]*100,2)}%, Banker={round(probs[1]*100,2)}%, Tie={round(probs[2]*100,2)}%")
    else:
        st.warning("⚠️ Cần ít nhất 10 kết quả để huấn luyện mô hình.")




# Dự đoán theo thống kê đơn giản
def baseline_prediction(data):
    if not data:
        return "Không có dữ liệu"
    return max(set(data), key=data.count)

# Giao diện nút dự đoán thống kê
if st.button("📊 Dự đoán theo tần suất"):
    baseline = baseline_prediction(st.session_state["data"])
    st.info(f"🔍 Dự đoán theo thống kê: {baseline}")


