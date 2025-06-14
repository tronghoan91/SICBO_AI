import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# Check python version
if sys.version_info < (3, 11) or sys.version_info >= (3, 13):
    print("Please use Python 3.11.x or 3.12.x for stability.")
    sys.exit(1)

load_dotenv()
BOT_TOKEN = os.getenv('BOT_TOKEN')
DATA_FILE = os.getenv('DATA_FILE', '/var/data/data.json')

# Load or initialize data
def load_data():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'w') as f:
            json.dump([], f)
    with open(DATA_FILE, 'r') as f:
        return json.load(f)

# Save data
def save_data(result):
    data = load_data()
    data.append(result)
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f)

# Streak detection
def detect_streak(data):
    tai_streak, xiu_streak = 0, 0
    for result in reversed(data):
        if sum(result) >= 11:
            tai_streak += 1
            xiu_streak = 0
        else:
            xiu_streak += 1
            tai_streak = 0
        if tai_streak >= 3:
            return "📌 Chuỗi Tài đang kéo dài, khả năng tiếp tục là cao."
        elif xiu_streak >= 3:
            return "📌 Chuỗi Xỉu đang kéo dài, khả năng tiếp tục là cao."
    return ""

# Train and predict with ensemble models
def predict_next():
    data = load_data()
    if len(data) < 5:
        return "Cần thêm dữ liệu để dự đoán chính xác hơn."

    df = pd.DataFrame(data, columns=['x1', 'x2', 'x3'])
    df['sum'] = df.sum(axis=1)
    df['tai_xiu'] = np.where(df['sum'] >= 11, 'Tài', 'Xỉu')
    df['chan_le'] = np.where(df['sum'] % 2 == 0, 'Chẵn', 'Lẻ')
    df['bao'] = np.where((df.x1 == df.x2) & (df.x2 == df.x3), 1, 0)

    X = df[['x1', 'x2', 'x3']]
    recent_result = np.array([data[-1]])

    clf_tai_xiu_rf = RandomForestClassifier().fit(X[:-1], df['tai_xiu'][:-1])
    clf_chan_le_rf = RandomForestClassifier().fit(X[:-1], df['chan_le'][:-1])
    clf_sum_gb = GradientBoostingClassifier().fit(X[:-1], df['sum'][:-1])
    clf_bao_rf = RandomForestClassifier().fit(X[:-1], df['bao'][:-1])

    tai_xiu_pred_rf = clf_tai_xiu_rf.predict_proba(recent_result)[0]
    chan_le_pred_rf = clf_chan_le_rf.predict_proba(recent_result)[0]
    sum_pred_gb = clf_sum_gb.predict_proba(recent_result)[0]
    bao_pred_rf = clf_bao_rf.predict_proba(recent_result)[0][1]

    tai_xiu = clf_tai_xiu_rf.classes_[np.argmax(tai_xiu_pred_rf)]
    chan_le = clf_chan_le_rf.classes_[np.argmax(chan_le_pred_rf)]

    sum_prob_sorted = sorted(zip(clf_sum_gb.classes_, sum_pred_gb), key=lambda x: x[1], reverse=True)
    top_sums = [str(s[0]) for s in sum_prob_sorted[:3]]

    storm_warning = "⚠️ Cảnh báo BÃO (bộ ba đồng nhất)!" if bao_pred_rf > 0.1 else ""
    streak_message = detect_streak(data)

    correct_tai_xiu = sum(clf_tai_xiu_rf.predict(X[:-1]) == df['tai_xiu'][:-1])
    total = len(X) - 1
    accuracy = round((correct_tai_xiu / total) * 100, 2)

    return (f"✨ Phiên sau dễ về: {tai_xiu} ({round(max(tai_xiu_pred_rf)*100)}%), {chan_le} ({round(max(chan_le_pred_rf)*100)}%)\n"
            f"🎯 Dải điểm dễ trúng: {', '.join(top_sums)}\n"
            f"🔮 BOT đã dự đoán trúng {correct_tai_xiu}/{total} phiên ({accuracy}%)\n"
            f"{storm_warning}\n{streak_message}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        result = [int(x) for x in update.message.text.strip()]
        if len(result) != 3 or any(d < 1 or d > 6 for d in result):
            raise ValueError

        save_data(result)
        prediction = predict_next()
        msg = f"🎲 Vừa ra: {result} → Tổng {sum(result)} ({'Tài' if sum(result)>=11 else 'Xỉu'}), {'Chẵn' if sum(result)%2==0 else 'Lẻ'}\n{prediction}"

        await update.message.reply_text(msg)

    except ValueError:
        await update.message.reply_text("Sai định dạng! Chỉ cần nhập 3 số xúc sắc (vd: 345)")

if __name__ == '__main__':
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.run_polling()
