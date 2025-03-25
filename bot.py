import telebot
from telebot import types
import pandas as pd
import datetime
import time
import requests
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
from model import load_data, train_model, predict_future

TOKEN = "7923521036:AAHcSpyCU-y8-YmRQQfOyx43ZfupNn-LFAc"
bot = telebot.TeleBot(TOKEN, threaded=True)

class States:
    WAITING_DATE = 1
    WAITING_PERIODS = 2

user_state = {}
predictions_data = {}

def init_model():
    try:
        print("Инициализация модели...")
        train, test = load_data()
        model, train_with_predictions = train_model(train)
        
        # Берем последнюю дату из данных ИЛИ текущую дату (что больше)
        last_known_date = test['date'].iloc[-1] if not test.empty else train['date'].iloc[-1]
        current_date = pd.to_datetime(datetime.datetime.now())
        last_date = max(last_known_date, current_date)
        
        print("Модель готова к работе!")
        return model, train_with_predictions, test, last_date
    except Exception as e:
        print(f"Ошибка инициализации модели: {e}")
        sys.exit(1)

def generate_plots(train_data, test_data, predictions, focus_date=None, periods=4):
    """Генерирует два графика: увеличенный и общий"""
    plt.figure(figsize=(14, 12))
    
    # 1. Увеличенный график (верхний)
    ax1 = plt.subplot(2, 1, 1)
    if focus_date:
        start_date = focus_date - pd.Timedelta(weeks=2)
        end_date = focus_date + pd.Timedelta(weeks=periods+2)
        
        # Фильтрация данных
        zoom_train = train_data[(train_data['date'] >= start_date) & 
                              (train_data['date'] <= end_date)]
        zoom_test = test_data[(test_data['date'] >= start_date) & 
                            (test_data['date'] <= end_date)] if test_data is not None else None
        zoom_pred = predictions[(predictions['date'] >= focus_date)]
        
        # Построение
        if not zoom_train.empty:
            ax1.plot(zoom_train['date'], zoom_train['price'], 'b-', label='Исторические данные')
        if zoom_test is not None and not zoom_test.empty:
            ax1.plot(zoom_test['date'], zoom_test['price'], 'b-')
        if not zoom_pred.empty:
            ax1.plot(zoom_pred['date'], zoom_pred['predicted_price'], 'r-o', 
                    linewidth=2, markersize=8, label='Прогноз')
        
        ax1.set_title(f"Увеличенный вид ({focus_date.strftime('%d.%m.%Y')} + {periods} недель)")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
        ax1.grid(True)
        ax1.legend()
    
    # 2. Общий график (нижний)
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(train_data['date'], train_data['price'], 'b-', label='Исторические данные')
    if test_data is not None:
        ax2.plot(test_data['date'], test_data['price'], 'b-')
    ax2.plot(predictions['date'], predictions['predicted_price'], 'r-o', 
            linewidth=2, markersize=6, label='Прогноз')
    
    ax2.set_title("Общий вид прогноза")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120)
    buf.seek(0)
    plt.close()
    return buf

@bot.message_handler(commands=['start'])
def send_welcome(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("Автопрогноз на 6 недель")
    btn2 = types.KeyboardButton("Сделать прогноз")
    markup.add(btn1, btn2)
    
    bot.send_message(
        message.chat.id,
        "🔮 Бот прогнозирования цен на арматуру\n\n"
        "Выберите действие:\n"
        "- 'Автопрогноз' - прогноз на 6 недель\n"
        "- 'Сделать прогноз' - ввести свою дату",
        reply_markup=markup
    )

@bot.message_handler(func=lambda m: m.text == "Автопрогноз на 6 недель")
def auto_predict(message):
    try:
        bot.send_chat_action(message.chat.id, 'typing')
        
        # Берем текущую дату, округленную до понедельника
        today = pd.to_datetime(datetime.datetime.now())
        next_monday = today + pd.Timedelta(days=(7 - today.weekday()))
        
        # Делаем прогноз от ближайшего понедельника
        predictions = predict_future(model, next_monday, periods=6)
        
        response = "📊 Прогноз на 6 недель с {}:\n\n".format(
            next_monday.strftime('%d.%m.%Y'))
        
        for _, row in predictions.iterrows():
            response += f"📅 {row['date'].strftime('%d.%m.%Y')}: {int(row['predicted_price']):,} руб.\n"
        
        plot_buf = generate_plots(train_data, test_data, predictions, next_monday, 6)
        bot.send_photo(message.chat.id, photo=plot_buf)
        bot.send_message(message.chat.id, response)
        
    except Exception as e:
        bot.send_message(message.chat.id, "⚠️ Ошибка прогноза. Попробуйте позже.")
        print(f"Ошибка автопрогноза: {e}")

@bot.message_handler(func=lambda m: m.text == "Сделать прогноз")
def start_custom_predict(message):
    msg = bot.send_message(
        message.chat.id,
        "📅 Введите дату начала прогноза (ДД.ММ.ГГГГ):"
    )
    user_state[message.chat.id] = States.WAITING_DATE
    bot.register_next_step_handler(msg, process_date)

def process_date(message):
    try:
        input_date = datetime.datetime.strptime(message.text, "%d.%m.%Y")
        predictions_data[message.chat.id] = {'date': input_date}
        
        msg = bot.send_message(
            message.chat.id,
            "⏳ Введите количество недель (1-12):"
        )
        user_state[message.chat.id] = States.WAITING_PERIODS
        bot.register_next_step_handler(msg, process_periods)
    except ValueError:
        bot.send_message(
            message.chat.id,
            "❌ Неверный формат даты. Введите ДД.ММ.ГГГГ:"
        )
        bot.register_next_step_handler(message, process_date)

def process_periods(message):
    try:
        periods = int(message.text)
        if not 1 <= periods <= 12:
            raise ValueError
        
        user_data = predictions_data[message.chat.id]
        bot.send_chat_action(message.chat.id, 'typing')
        predictions = predict_future(model, user_data['date'], periods)
        
        response = f"📊 Прогноз на {periods} недель с {user_data['date'].strftime('%d.%m.%Y')}:\n\n"
        for _, row in predictions.iterrows():
            response += f"📅 {row['date'].strftime('%d.%m.%Y')}: {int(row['predicted_price']):,} руб.\n"
        
        plot_buf = generate_plots(
            train_data, 
            test_data, 
            predictions,
            focus_date=user_data['date'],
            periods=periods
        )
        
        bot.send_photo(message.chat.id, photo=plot_buf)
        bot.send_message(message.chat.id, response)
        user_state.pop(message.chat.id, None)
    except ValueError:
        bot.send_message(
            message.chat.id,
            "❌ Введите число от 1 до 12:"
        )
        bot.register_next_step_handler(message, process_periods)
    except Exception as e:
        bot.send_message(message.chat.id, "⚠️ Ошибка расчета.")
        print(f"Ошибка прогноза: {e}")
        user_state.pop(message.chat.id, None)

def start_bot():
    while True:
        try:
            print("Бот запущен...")
            bot.polling(none_stop=True, interval=3, timeout=30)
        except requests.exceptions.ReadTimeout:
            print("Переподключение через 5 сек...")
            time.sleep(5)
        except Exception as e:
            print(f"Ошибка: {e}, перезапуск через 15 сек...")
            time.sleep(15)

if __name__ == "__main__":
    model, train_data, test_data, last_date = init_model()
    start_bot()