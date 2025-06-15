import os
import numpy as np
import pandas as pd
from flask import Flask, request
from telegram import Update, Bot
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
import logging

from sqlalchemy import create_engine, Column, Integer, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

import tensorflow as tf
from tensorflow.keras import layers, models

# ENV CONFIG
BOT_TOKEN = os.environ.get('BOT_TOKEN')
WEBHOOK_URL = os.environ.get('WEBHOOK_URL')
PORT = int(os.environ.get('PORT', 10000))
DATABASE_URL = os.environ.get('DATABASE_URL')

# == DATABASE SETUP ==
Base = declarative_base()

class SicboResult(Base):
    __tablename__ = 'sicbo_results'
    id = Column(Integer, primary_key=True, autoincrement=True)
    x1 = Column(Integer, nullable=False)
    x2 = Column(Integer, nullable=False)
    x3 = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# == SICBO LOGIC ==
def load_data(n=200):
    session = Session()
    # Lấy n kết quả gần nhất
    rows = session.query(SicboResult).order_by(SicboResult.id.desc()).limit(n).all()
    data = [[r.x1, r.x2, r.x3] for r in reversed(rows)]
    session.close()
    return data

def save_data(result):
    session = Session()
    new_row = SicboResult(x1=result[0], x2=result[1], x3=result[2])
    session.add(new_row)
    session.commit()
    session.close()

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
            return "🔎 Chuỗi Tài liên tục, đề phòng đảo chiều!"
        elif xiu_streak >= 3:
            return "🔎 Chuỗi Xỉu liên tục, đề phòng đảo chiều!"
    return ""

def prepare_lstm_data(data, window=5):
    X, y = [], []
    sums = [sum(x) for x in data]
    targets = [1 if s >= 11 else 0 for s in sums]
    for i in range(len(targets)-window):
        X.append(targets[i:i+window])
        y.append(targets[i+window])
    X = np.array(X)
    y = np.array(y)
    return X, y

def train_lstm_model(data, window=5):
    X, y = prepare_lstm_data(data, window)
    if len(X) < 10:
        return None
    model = models.Sequential([
        layers.Input((X.shape[1], 1)),
        layers.LSTM(16, return_sequences=False),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X[..., np.newaxis], y, epochs=10, verbose=0)
    return model

def predict_lstm_next(data, model, window=5):
    sums = [sum(x) for x in data]
    last = np.array(sums[-window:])
    inp = (last >= 11).astype(int).reshape((1, window, 1))
    prob = model.predict(inp)[0][0]
    return 'Tài' if prob > 0.5 else 'Xỉu', prob

def analyze_history(df, n=20):
    stats = {}
    stats['tai'] = int(sum(df['tai_xiu'][-n:] == 'Tài'))
    stats['xiu'] = n - stats['tai']
    stats['chan'] = int(sum(df['chan_le'][-n:] == 'Chẵn'))
    stats['le'] = n - stats['chan']
    return stats

def weighted_vote(*probs, weights=None):
    if not weights:
        weights = [1/len(probs)] * len(probs)
    return sum([p*w for p, w in zip(probs, weights)])

def predict_next(result, data=None):
    if data is None:
        data = load_data()
    if len(data) < 10:
        return "Cần thêm dữ liệu để dự đoán chính xác hơn."
    df = pd.DataFrame(data, columns=['x1', 'x2', 'x3'])
    df['sum'] = df.sum(axis=1)
    df['tai_xiu'] = np.where(df['sum'] >= 11, 'Tài', 'Xỉu')
    df['chan_le'] = np.where(df['sum'] % 2 == 0, 'Chẵn', 'Lẻ')
    df['bao'] = np.where((df.x1 == df.x2) & (df.x2 == df.x3), 1, 0)
    X = df[['x1', 'x2', 'x3']]
    y_tai_xiu = df['tai_xiu']

    clf_tai_xiu_rf = RandomForestClassifier().fit(X[:-1], y_tai_xiu[:-1])
    clf_chan_le_rf = RandomForestClassifier().fit(X[:-1], df['chan_le'][:-1])
    clf_sum_gb = GradientBoostingClassifier().fit(X[:-1], df['sum'][:-1])
    clf_bao_rf = RandomForestClassifier().fit(X[:-1], df['bao'][:-1])
    stack = StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier()),
            ('gb', GradientBoostingClassifier())
        ],
        final_estimator=LogisticRegression(),
        n_jobs=1
    )
    stack.fit(X[:-1], (df['tai_xiu'][:-1] == 'Tài').astype(int))
    clf_nb = GaussianNB()
    clf_nb.fit(X[:-1], (df['tai_xiu'][:-1] == 'Tài').astype(int))
    recent_result = np.array([data[-1]])
    tai_xiu_pred_rf = clf_tai_xiu_rf.predict_proba(recent_result)[0]
    chan_le_pred_rf = clf_chan_le_rf.predict_proba(recent_result)[0]
    sum_pred_gb = clf_sum_gb.predict_proba(recent_result)[0]
    bao_pred_rf = clf_bao_rf.predict_proba(recent_result)[0][1]
    stack_pred = stack.predict_proba(recent_result)[0][1]
    nb_pred_prob = clf_nb.predict_proba(recent_result)[0][1]

    tai_xiu = clf_tai_xiu_rf.classes_[np.argmax(tai_xiu_pred_rf)]
    weighted_pred = weighted_vote(
        tai_xiu_pred_rf[np.argmax(clf_tai_xiu_rf.classes_ == 'Tài')],
        stack_pred, nb_pred_prob)
    weighted_label = 'Tài' if weighted_pred > 0.5 else 'Xỉu'
    sum_prob_sorted = sorted(zip(clf_sum_gb.classes_, sum_pred_gb), key=lambda x: x[1], reverse=True)
    top_sums = [str(s[0]) for s in sum_prob_sorted[:3]]

    last_stats = analyze_history(df, n=20)
    streak_message = detect_streak(data)
    alert = ""
    if abs(last_stats['tai']/20 - 0.5) > 0.15:
        alert = f"🚨 Lưu ý: {'Tài' if last_stats['tai'] > last_stats['xiu'] else 'Xỉu'} xuất hiện nhiều hơn 15% so với lý thuyết!"

    suggest = ""
    if weighted_pred > 0.7:
        suggest = f"⚡ Khuyến nghị: Có thể cược mạnh cửa {weighted_label}."
    elif weighted_pred > 0.55:
        suggest = f"⚡ Khuyến nghị: Cược {weighted_label}, kiểm soát vốn."
    else:
        suggest = "⚡ Khuyến nghị: Dữ liệu phân tán, chỉ cược nhẹ!"

    correct_tai_xiu = sum(clf_tai_xiu_rf.predict(X[:-1]) == df['tai_xiu'][:-1])
    total = len(X) - 1
    accuracy = round((correct_tai_xiu / total) * 100, 2)
    thong_ke = f"🔮 BOT đã dự đoán đúng {correct_tai_xiu}/{total} phiên ({accuracy}%)"

    # LSTM
    lstm_msg = ""
    if len(data) >= 20:
        try:
            lstm_model = train_lstm_model(data)
            if lstm_model:
                lstm_pred, lstm_prob = predict_lstm_next(data, lstm_model)
                lstm_msg = f"🧠 LSTM: {lstm_pred} ({round(lstm_prob*100,1)}%)"
        except Exception as e:
            lstm_msg = ""

    line1 = f"🎲 Vừa ra: {result} → Tổng {sum(result)} ({'Tài' if sum(result)>=11 else 'Xỉu'}, {'Chẵn' if sum(result)%2==0 else 'Lẻ'})"
    line2 = f"✨ Dự đoán mạnh: {weighted_label} ({round(weighted_pred*100,1)}%) | Dải điểm: {', '.join(top_sums)}"
    line3 = f"📈 20 phiên gần nhất: Tài {last_stats['tai']*5}%, Xỉu {last_stats['xiu']*5}%"
    extra = "\n".join(filter(None, [alert, streak_message, suggest, lstm_msg, thong_ke]))
    return "\n".join([line1, line2, line3, extra]).strip()

# == BOT & FLASK APP ==
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
bot = Bot(BOT_TOKEN)
application = ApplicationBuilder().token(BOT_TOKEN).build()
dispatcher = application.dispatcher

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        result = [int(x) for x in update.message.text.strip()]
        if len(result) != 3 or any(d < 1 or d > 6 for d in result):
            raise ValueError
        save_data(result)
        data = load_data()
        msg = predict_next(result, data)
        await update.message.reply_text(msg)
    except Exception:
        await update.message.reply_text("Sai định dạng! Chỉ cần nhập 3 số xúc sắc (vd: 345)")

dispatcher.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

@app.route('/')
def home():
    return "Sicbo BOT hoạt động OK!", 200

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json(force=True)
    update = Update.de_json(data, bot)
    application.process_update(update)
    return 'OK'

if __name__ == "__main__":
    bot.delete_webhook()
    bot.set_webhook(url=WEBHOOK_URL)
    app.run(host='0.0.0.0', port=PORT)
