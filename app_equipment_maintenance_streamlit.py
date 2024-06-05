import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import matplotlib.pyplot as plt

# Генератор текущих значений датчиков
def generate_current_sensor_values():
    return {col: np.random.uniform(low=0, high=15) for col in sensor_cols}

# Функция обновления данных на дашборде
def update_dashboard():
    # Обновление текущих значений датчиков
    current_values = generate_current_sensor_values()
    
    # Отображение текущих значений датчиков
    with current_values_placeholder.container():
        st.subheader("Current Sensor Values")
        gauge_figures = []
        
        # Pressure
        gauge_figures.append(go.Indicator(
            mode="gauge+number",
            value=current_values['P3-111/A_PT_341'],
            title={'text': "Pressure"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [0, 15]}, 'bar': {'color': 'lightblue'}}
        ))
        
        # Temperature
        gauge_figures.append(go.Indicator(
            mode="gauge+number",
            value=current_values['P3-111/A_TT_341'],
            title={'text': "Temperature"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [0, 15]}, 'bar': {'color': 'coral'}}
        ))

        # Vibration Motor
        gauge_figures.append(go.Indicator(
            mode="gauge+number",
            value=current_values['P3-111/A_VT_341'],
            title={'text': "Vibration Motor"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [0, 15]}, 'bar': {'color': 'navy'}}
        ))

        # Vibration Pump
        gauge_figures.append(go.Indicator(
            mode="gauge+number",
            value=current_values['P3-111/A_VT_342'],
            title={'text': "Vibration Pump"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [0, 15]}, 'bar': {'color': 'dodgerblue'}}
        ))

        # Vibration Unit
        gauge_figures.append(go.Indicator(
            mode="gauge+number",
            value=current_values['P3-111/A_VT_343'],
            title={'text': "Vibration Unit"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [0, 15]}, 'bar': {'color': 'steelblue'}}
        ))

        # Рисуем датчики в два ряда
        fig = make_subplots(rows=2, cols=3, specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                                                    [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]])

        fig.add_trace(gauge_figures[0], row=1, col=1)
        fig.add_trace(gauge_figures[1], row=1, col=2)
        fig.add_trace(gauge_figures[2], row=1, col=3)
        fig.add_trace(gauge_figures[3], row=2, col=1)
        fig.add_trace(gauge_figures[4], row=2, col=2)
        
        st.plotly_chart(fig)

    # Добавление новых точек на графики
    current_time = pd.Timestamp.now()
    for col in sensor_cols:
        historical_data[col].append((current_time, current_values[col]))

    # Отображение исторических данных
    with historical_graphs_placeholder.container():
        st.subheader("Current Sensor Data")
        for col in sensor_cols:
            times, values = zip(*historical_data[col])
            fig = go.Figure(go.Scatter(x=list(times), y=list(values), mode='lines', name=col, line=dict(color=colors[col])))
            fig.update_layout(title=f"Current Data for {col}",
                              xaxis_title="Time",
                              yaxis_title="Value",
                              showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

    # Прогнозы и анализ
    with predictions_analysis_placeholder.container():
        st.subheader("Predictions and Analysis")
        for col in sensor_cols:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sarima_forecasts.index, y=sarima_forecasts[col], mode='lines', name='Forecast', line=dict(color=colors[col])))
            fig.add_trace(go.Scatter(x=data.index, y=data[col], mode='lines', name='Actual', line=dict(color=colors[col], dash='dash')))
            fig.update_layout(title=f"Forecast vs Actual for {col}",
                              xaxis_title="Time",
                              yaxis_title="Value",
                              showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
    # Визуализация неисправностей
    with failures_graph_placeholder.container():
        st.subheader("Diagnosed Failures")
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(combined_data.index, combined_data['P3-111/A_PT_341'], color='lightblue', label='Pressure (P3-111/A_PT_341)')
        ax.plot(combined_data.index, combined_data['P3-111/A_TT_341'], color='coral', label='Temperature (P3-111/A_TT_341)')
        ax.plot(combined_data.index, combined_data['P3-111/A_VT_341'], color='navy', label='Vibration Motor (P3-111/A_VT_341)')
        ax.plot(combined_data.index, combined_data['P3-111/A_VT_342'], color='dodgerblue', label='Vibration Pump (P3-111/A_VT_342)')
        ax.plot(combined_data.index, combined_data['P3-111/A_VT_343'], color='steelblue', label='Vibration Unit (P3-111/A_VT_343)')

        for failure in classifications:
            if failure[0] != 'No issue':
                ax.scatter(failure[2], combined_data.loc[failure[2], sensor_cols].max().max(), color='red', marker='o', s=100, zorder=5, label=f'{int(failure[3])}')

        ax.set_xlabel('Time')
        ax.set_ylabel('Sensor Readings')
        ax.set_title('Diagnosed Failures from 27 August 2023 to 31 March 2024')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

    # Визуализация сводки
    with summary_placeholder.container():
        st.subheader("Сводка по насосу P3-111А")
        st.table(wear_df[['month', 'wear', 'residual_life']])

        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(wear_df['month'], wear_df['wear'], color='blue', label='Износ (в %)')
        ax.plot(wear_df['month'], wear_df['residual_life'], color='orange', label='Остаточный ресурс (в %)')
        ax.set_xlabel('Месяцы')
        ax.set_ylabel('Процент')
        ax.set_title('Износ и остаточный ресурс насоса 111А')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

    # Отображение предупреждений
    with warnings_placeholder.container():
        st.subheader("Diagnosed Failures")
        warning_list = ["<li>{}. {}</li>".format(int(c[3]), c[0]) for c in classifications if c[0] != 'No issue']
        warning_message = "<ul>{}</ul>".format("".join(warning_list))
        full_warning_message = "<div style='font-size:18px;'><strong>Внимание: Есть риск неисправностей:</strong><br>{}</div>".format(warning_message)
        st.markdown(full_warning_message, unsafe_allow_html=True)

# Настройка страницы
st.set_page_config(layout="wide")

# Данные
sensor_cols = ['P3-111/A_PT_341', 'P3-111/A_TT_341', 'P3-111/A_VT_341', 'P3-111/A_VT_342', 'P3-111/A_VT_343']
colors = {
    'P3-111/A_PT_341': 'lightblue',
    'P3-111/A_TT_341': 'coral',
    'P3-111/A_VT_341': 'navy',
    'P3-111/A_VT_342': 'dodgerblue',
    'P3-111/A_VT_343': 'steelblue'
}
historical_data = {col: [] for col in sensor_cols}

# Загрузка обновленных прогнозов SARIMA с шумами
updated_sarima_forecasts_path = 'C:\\Users\\Admin\\Desktop\\Питон\\Nordal\\P-111\\updated_sarima_forecasts.csv'
sarima_forecasts = pd.read_csv(updated_sarima_forecasts_path, parse_dates=['time'], dayfirst=True)
sarima_forecasts.set_index('time', inplace=True)

data_path = 'C:\\Users\\Admin\\Desktop\\Питон\\Nordal\\P-111\\modified_filled_data_24.05.31.csv'
data = pd.read_csv(data_path, parse_dates=['time'])
data['time'] = pd.to_datetime(data['time'])
data.set_index('time', inplace=True)

combined_data = pd.concat([data[sensor_cols], sarima_forecasts[sensor_cols]])
monthly_data = combined_data.resample('M').mean()
base_period = monthly_data.loc[(monthly_data.index >= '2023-09-01') & (monthly_data.index <= '2023-09-30')]
base_means = base_period.mean()
percentage_changes = (monthly_data - base_means) / base_means * 100

wear_data_path = r'C:\Users\Admin\Desktop\Питон\Nordal\P-111\wear_and_residual_life.csv'
wear_df = pd.read_csv(wear_data_path, parse_dates=['month'])
wear_df['month'] = pd.to_datetime(wear_df['month'])

failures_dict_path = r'C:\Users\Admin\Desktop\Питон\Nordal\\P-111\\failures_dict_2024_06_02.csv'
failures_dict = pd.read_csv(failures_dict_path)
failures_dict = failures_dict[failures_dict['duration'] != 0.001]

# Функция классификации неисправностей
def classify_issues(percentage_changes, monthly_means, failures_dict, df):
    classifications = []
    for _, failure in failures_dict.iterrows():
        tt_duration = calculate_duration(df.loc['2023-10-01':'2024-03-31'], 'P3-111/A_TT_341', failure['tt_changes'])
        vt_motor_duration = calculate_duration(df.loc['2023-10-01':'2024-03-31'], 'P3-111/A_VT_341', failure['vt_motor_changes_critical'])
        vt_pump_duration = calculate_duration(df.loc['2023-10-01':'2024-03-31'], 'P3-111/A_VT_343', failure['vt_pump_changes_critical'])
        
        if (
            (percentage_changes['P3-111/A_TT_341'].iloc[-1] >= failure['tt_changes']) and
            (percentage_changes['P3-111/A_VT_341'].iloc[-1] >= failure['vt_motor_changes_critical']) and
            (percentage_changes['P3-111/A_VT_342'].iloc[-1] >= failure['vt_motor_changes_critical']) and
            (percentage_changes['P3-111/A_VT_343'].iloc[-1] >= failure['vt_pump_changes_critical']) and
            (tt_duration >= failure['duration']) and
            (vt_motor_duration >= failure['duration']) and
            (vt_pump_duration >= failure['duration'])
        ):
            classifications.append((failure['pumps_falure'], failure['duration'], df.index[np.argmax(df[sensor_cols].max())], failure['failure_number']))
    return classifications if classifications else [('No issue', None, None, None)]

# Функция для определения продолжительности превышений
def calculate_duration(data, sensor_col, threshold):
    duration = 0
    max_duration = 0
    for value in data[sensor_col]:
        if value >= threshold:
            duration += 1
        else:
            max_duration = max(max_duration, duration)
            duration = 0
    max_duration = max(max_duration, duration)
    return max_duration

classifications = classify_issues(percentage_changes, monthly_data, failures_dict, combined_data)

# Место для виджетов
current_values_placeholder = st.empty()
historical_graphs_placeholder = st.empty()
predictions_analysis_placeholder = st.empty()
failures_graph_placeholder = st.empty()
warnings_placeholder = st.empty()
summary_placeholder = st.empty()

# Кнопка для остановки генерации данных
stop_button = st.button("Stop Generation")
if stop_button:
    st.stop()

# Запуск обновления данных каждые 2 секунды
while True:
    update_dashboard()
    time.sleep(2)
