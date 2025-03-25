# 🏗️ Прогнозирование цен на арматуру с помощью Telegram-бота

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange?logo=scikit-learn)
![Telegram](https://img.shields.io/badge/Telegram-Bot-blue?logo=telegram)

## 📌 О проекте

Проект представляет собой интеллектуальную систему прогнозирования цен на строительную арматуру с интеграцией в Telegram. 

**Ключевые особенности:**
- Модель машинного обучения (Random Forest) для прогнозирования цен
- Двухпанельная визуализация результатов
- Гибкий выбор периода прогнозирования (1-12 недель)
- Автоматическое обновление прогноза от текущей даты

## 📊 Архитектура решения

```mermaid
graph TD
    A[Telegram Bot] --> B[Модель Random Forest]
    B --> C[Данные цен на арматуру]
    C --> D[Графики прогноза]
    D --> E[Пользователь]