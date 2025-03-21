import pandas as pd
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# Sample data preparation
data = {
    'transaction_amount': [2500, 799, 9338, 11749, 8999, 1500, 3000, 4000, 300, 5000, 24990],
    'transaction_date': ['01-11-2024 16:08', '01-11-2024 16:15', '02-11-2024 14:43', '03-11-2024 11:14', 
                         '04-11-2024 12:54', '06-11-2024 08:36', '06-11-2024 08:56', '06-11-2024 09:08', 
                         '06-11-2024 09:29', '06-11-2024 13:05', '06-11-2024 15:12'],
    'transaction_channel': ['mobile', 'mobile', 'mobile', 'mobile', 'mobile', 'W', 'W', 'W', 'W', 'W', 'mobile'],
    'is_fraud': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'transaction_payment_mode_anonymous': [10, 10, 2, 6, 2, 10, 10, 10, 10, 10, 2],
    'payment_gateway_bank_anonymous': [6, 6, 6, 58, 6, 6, 6, 6, 6, 6, 6],
    'payer_browser_anonymous': [1833, 1833, 2766, 3378, 2766, 3212, 3212, 3212, 3212, 3212, 2721],
    'transaction_id_anonymous': ['ANON_9629', 'ANON_9764', 'ANON_27514', 'ANON_41176', 'ANON_66597', 
                                'ANON_134329', 'ANON_134618', 'ANON_134815', 'ANON_135218', 
                                'ANON_147464', 'ANON_155578'],
    'payee_id_anonymous': ['ANON_47', 'ANON_47', 'ANON_265', 'ANON_8', 'ANON_265', 'ANON_12', 
                          'ANON_12', 'ANON_12', 'ANON_12', 'ANON_12', 'ANON_265']
}

df = pd.DataFrame(data)

df['transaction_date'] = pd.to_datetime(df['transaction_date'], format='%d-%m-%Y %H:%M')

np.random.seed(42)
df['is_fraud_predicted'] = np.random.choice([0, 1], size=len(df), p=[0.3, 0.7])
df['is_fraud_reported'] = np.random.choice([0, 1], size=len(df), p=[0.4, 0.6])

def filter_data(start_date, end_date, payer_id, payee_id, transaction_id):
    filtered_df = df.copy()
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    filtered_df = filtered_df[(filtered_df['transaction_date'] >= start_date) & 
                             (filtered_df['transaction_date'] <= end_date)]
    
    if payer_id:
        filtered_df = filtered_df[filtered_df['transaction_id_anonymous'] == payer_id]
    
    if payee_id:
        filtered_df = filtered_df[filtered_df['payee_id_anonymous'] == payee_id]
    
    if transaction_id:
        filtered_df = filtered_df[filtered_df['transaction_id_anonymous'] == transaction_id]
    
    return filtered_df

def create_comparison_chart(dimension, filtered_df):
    if filtered_df.empty:
        return plt.figure()
    
    plt.figure(figsize=(10, 6))
    
    if dimension == 'Transaction Channel':
        group_col = 'transaction_channel'
    elif dimension == 'Transaction Payment Mode':
        group_col = 'transaction_payment_mode_anonymous'
    elif dimension == 'Payment Gateway Bank':
        group_col = 'payment_gateway_bank_anonymous'
    elif dimension == 'Payer ID':
        group_col = 'transaction_id_anonymous'
    elif dimension == 'Payee ID':
        group_col = 'payee_id_anonymous'
    else:
        return plt.figure()
    
    predicted = filtered_df.groupby(group_col)['is_fraud_predicted'].sum()
    reported = filtered_df.groupby(group_col)['is_fraud_reported'].sum()
    
    plot_df = pd.DataFrame({
        'Predicted Fraud': predicted,
        'Reported Fraud': reported
    })
    
    plot_df.plot(kind='bar', figsize=(10, 6))
    plt.title(f'Fraud Comparison by {dimension}')
    plt.ylabel('Count')
    plt.xlabel(dimension)
    plt.tight_layout()
    
    return plt

def create_time_series(filtered_df, granularity):
    if filtered_df.empty:
        return plt.figure()
    
    plt.figure(figsize=(12, 6))
    
    if granularity == 'Day':
        time_group = filtered_df['transaction_date'].dt.date
    elif granularity == 'Hour':
        time_group = filtered_df['transaction_date'].dt.strftime('%Y-%m-%d %H')
    elif granularity == 'Minute':
        time_group = filtered_df['transaction_date'].dt.strftime('%Y-%m-%d %H:%M')
    else:
        return plt.figure()
    
    predicted = filtered_df.groupby(time_group)['is_fraud_predicted'].sum()
    reported = filtered_df.groupby(time_group)['is_fraud_reported'].sum()
    
    plt.plot(predicted.index, predicted.values, 'b-', label='Predicted Fraud')
    plt.plot(reported.index, reported.values, 'r-', label='Reported Fraud')
    plt.title('Fraud Trend Over Time')
    plt.ylabel('Count')
    plt.xlabel('Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return plt

def calculate_metrics(filtered_df):
    if filtered_df.empty:
        return None, 0, 0
    
    cm = confusion_matrix(filtered_df['is_fraud'], filtered_df['is_fraud_predicted'])
    
    precision = precision_score(filtered_df['is_fraud'], filtered_df['is_fraud_predicted'], zero_division=0)
    recall = recall_score(filtered_df['is_fraud'], filtered_df['is_fraud_predicted'], zero_division=0)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Fraud', 'Fraud'],
                yticklabels=['Not Fraud', 'Fraud'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    return plt, precision, recall

def update_interface(start_date, end_date, payer_id, payee_id, transaction_id, dimension, time_granularity):
    filtered_df = filter_data(start_date, end_date, payer_id, payee_id, transaction_id)
    
    comparison_chart = create_comparison_chart(dimension, filtered_df)
    
    time_series = create_time_series(filtered_df, time_granularity)
    
    confusion_matrix_plot, precision, recall = calculate_metrics(filtered_df)
    
    display_df = filtered_df.copy()
    display_df['transaction_date'] = display_df['transaction_date'].dt.strftime('%Y-%m-%d %H:%M')
    
    return (display_df.to_dict('records'), 
            comparison_chart, 
            time_series, 
            confusion_matrix_plot, 
            f"Precision: {precision:.4f}", 
            f"Recall: {recall:.4f}")

with gr.Blocks() as demo:
    gr.Markdown("# Fraud Transaction Analysis Dashboard")
    
    with gr.Row():
        with gr.Column():
            start_date = gr.Textbox(label="Start Date (YYYY-MM-DD)", value="2024-11-01")
            end_date = gr.Textbox(label="End Date (YYYY-MM-DD)", value="2024-11-06")
        
        with gr.Column():
            payer_id = gr.Textbox(label="Payer ID")
            payee_id = gr.Textbox(label="Payee ID")
            transaction_id = gr.Textbox(label="Transaction ID")
    
    with gr.Row():
        dimension = gr.Dropdown(
            ["Transaction Channel", "Transaction Payment Mode", "Payment Gateway Bank", "Payer ID", "Payee ID"],
            label="Comparison Dimension",
            value="Transaction Channel"
        )
        time_granularity = gr.Dropdown(
            ["Day", "Hour", "Minute"],
            label="Time Granularity",
            value="Day"
        )
    
    update_button = gr.Button("Update Dashboard")
    
    with gr.Row():
        gr.Markdown("## Transaction Data")
    
    data_table = gr.DataFrame()
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Fraud Comparison by Dimension")
            comparison_plot = gr.Plot()
        
        with gr.Column():
            gr.Markdown("## Fraud Trend Over Time")
            time_series_plot = gr.Plot()
    
    with gr.Row():
        gr.Markdown("## Model Evaluation")
    
    with gr.Row():
        with gr.Column():
            confusion_matrix_plot = gr.Plot()
        
        with gr.Column():
            precision_text = gr.Textbox(label="Precision")
            recall_text = gr.Textbox(label="Recall")
    
    update_button.click(
        update_interface,
        inputs=[start_date, end_date, payer_id, payee_id, transaction_id, dimension, time_granularity],
        outputs=[data_table, comparison_plot, time_series_plot, confusion_matrix_plot, precision_text, recall_text]
    )

if __name__ == "__main__":
    demo.launch()
