import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib
import os
import matplotlib.pyplot as plt
import io
from matplotlib.dates import YearLocator, DateFormatter
import matplotlib.dates as mdates

MODEL_PATH = 'armature_price_model.joblib'

def load_data(train_path='train.xlsx', test_path='test.xlsx'):
    """Загрузка и подготовка данных."""
    train = pd.read_excel(train_path)
    test = pd.read_excel(test_path)
    
    train = train.rename(columns={'dt': 'date', 'Цена на арматуру': 'price'})
    test = test.rename(columns={'dt': 'date', 'Цена на арматуру': 'price'})
    
    for df in [train, test]:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['week'] = df['date'].dt.isocalendar().week
        df['day_of_week'] = df['date'].dt.dayofweek
    
    return train, test

def train_model(train):
    """Обучение модели с кэшированием."""
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        train['prediction'] = model.predict(train[['year', 'month', 'week', 'day_of_week']])
        return model, train
    
    X = train[['year', 'month', 'week', 'day_of_week']]
    y = train['price']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    
    train['prediction'] = model.predict(X)
    
    val_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_pred)
    print(f"MAE на валидации: {mae:.2f} руб.")
    
    joblib.dump(model, MODEL_PATH)
    return model, train

def predict_future(model, last_date, periods=4):
    """Прогнозирование на будущие периоды."""
    future_dates = pd.date_range(start=last_date, periods=periods + 1, freq='W-MON')[1:]
    future_df = pd.DataFrame({
        'date': future_dates,
        'year': future_dates.year,
        'month': future_dates.month,
        'week': future_dates.isocalendar().week,
        'day_of_week': future_dates.dayofweek
    })
    predictions = model.predict(future_df[['year', 'month', 'week', 'day_of_week']])
    return future_df.assign(predicted_price=predictions)

def plot_predictions(train_data, test_data, predictions, focus_date=None, focus_weeks=4):
    """Создает два графика: увеличенный вид и общий вид"""
    # Используем стандартный стиль
    plt.style.use('default')
    
    # Создаем фигуру
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Настраиваем общий вид графиков
    for ax in [ax1, ax2]:
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlabel('Дата')
        ax.set_ylabel('Цена (руб)')
    
    # Верхний график (увеличенный вид)
    if focus_date:
        plot_zoomed_graph(ax1, train_data, test_data, predictions, focus_date, focus_weeks)
    else:
        plot_full_graph(ax1, train_data, test_data, predictions)
        ax1.set_title("Общий вид (без увеличения)")
    
    # Нижний график (полный вид)
    plot_full_graph(ax2, train_data, test_data, predictions)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf

def plot_full_graph(ax, train_data, test_data, predictions):
    """Рисует полный график на указанных осях"""
    # Исторические данные
    ax.plot(train_data['date'], train_data['price'], 
            label='Исторические данные', color='blue', linewidth=1.5)
    
    # Тестовые данные (если есть)
    if test_data is not None:
        ax.plot(test_data['date'], test_data['price'], 
                color='blue', linewidth=1.5)
    
    # Прогноз
    ax.plot(predictions['date'], predictions['predicted_price'], 
            label='Прогноз', color='orange', linestyle='-', 
            linewidth=2.5, marker='o', markersize=6)
    
    ax.set_title("Общий вид прогноза цен на арматуру", fontsize=12)
    ax.set_xlabel('Дата')
    ax.set_ylabel('Цена (руб)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    # Форматирование дат
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

def plot_zoomed_graph(ax, train_data, test_data, predictions, 
                     focus_date, focus_weeks):
    """Рисует увеличенный график выбранного периода"""
    # Определяем диапазон дат для увеличения
    start_date = focus_date - pd.Timedelta(weeks=2)  # Добавляем 2 недели до
    end_date = focus_date + pd.Timedelta(weeks=focus_weeks+2)  # И после
    
    # Фильтруем данные
    zoom_train = train_data[(train_data['date'] >= start_date) & 
                           (train_data['date'] <= end_date)]
    zoom_test = test_data[(test_data['date'] >= start_date) & 
                         (test_data['date'] <= end_date)] if test_data is not None else None
    zoom_pred = predictions[(predictions['date'] >= focus_date) & 
                           (predictions['date'] <= end_date)]
    
    # Рисуем
    if not zoom_train.empty:
        ax.plot(zoom_train['date'], zoom_train['price'], 
                label='Исторические данные', color='blue', linewidth=2)
    
    if zoom_test is not None and not zoom_test.empty:
        ax.plot(zoom_test['date'], zoom_test['price'], 
                color='blue', linewidth=2)
    
    if not zoom_pred.empty:
        ax.plot(zoom_pred['date'], zoom_pred['predicted_price'], 
                label='Прогноз', color='orange', linestyle='-', 
                linewidth=3, marker='o', markersize=8)
    
    ax.set_title(f"Увеличенный вид прогноза ({focus_date.strftime('%d.%m.%Y')} + {focus_weeks} недель)", 
                 fontsize=12)
    ax.set_xlabel('Дата')
    ax.set_ylabel('Цена (руб)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    # Форматирование дат для увеличенного графика
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
