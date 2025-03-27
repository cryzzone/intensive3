import pandas as pd
import tkinter as tk
from tkinter import messagebox
from darts import TimeSeries
from darts.models import ExponentialSmoothing
import matplotlib.pyplot as plt

forecast = pd.read_csv('forecast.csv', index_col=0, parse_dates=True)

def get_purchase_recommendation(input_date):
    try:
        input_date = pd.to_datetime(input_date)

        forecast_index = forecast.time_index
        nearest_index = forecast_index.get_indexer([input_date], method='nearest')[0]

        recommendation = forecast.values()[nearest_index]

        messagebox.showinfo("Рекомендация", f"Рекомендуемая цена на {input_date.date()}: {recommendation:.2f}")
    except Exception as e:
        messagebox.showerror("Ошибка", str(e))

root = tk.Tk()
root.title("Рекомендация по покупке")

tk.Label(root, text="Введите дату (YYYY-MM-DD):").pack(pady=10)
date_entry = tk.Entry(root)
date_entry.pack(pady=10)

tk.Button(root, text="Получить рекомендацию", command=lambda: get_purchase_recommendation(date_entry.get())).pack(pady=20)

root.mainloop()