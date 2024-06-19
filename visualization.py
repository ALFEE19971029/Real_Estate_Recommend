import pandas as pd
import requests
import pandas as pd
import numpy as np
import os
import xmltodict
import re
import time
import random
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def visualization(df):
    # Create subplots
    print(df)
    if len(df) == 0:
        fig, ax = plt.subplots()
        # 그래프 부분을 없애기 위해 축을 숨김
        ax.axis('off')
        # 제목을 설정하고 제목만 보이도록
        ax.set_title("No Apartment", fontsize=20, pad=20)
        # 그래프의 전체 크기를 제목으로 채우기
        plt.subplots_adjust(top=0.85)
        return plt

    fig = make_subplots(rows=1, cols=2, subplot_titles=("가격 비교", "상승률 비교"))

    # Add first subplot - Prices
    fig.add_trace(go.Bar(
        x=df['AptName_EArea2'],
        y=df['202212_real'],
        name='2022년 12월(실제)',
        marker_color='#4285F4',
        offsetgroup=0
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df['AptName_EArea2'],
        y=df['202312_predict'],
        name='2023년 12월(예측)',
        marker_color='#34A853',
        offsetgroup=1
    ), row=1, col=1)

    # Add second subplot - Growth rates
    fig.add_trace(go.Bar(
        x=df['AptName_EArea2'],
        y=df['202312_predict']/df['202212_real'],
        name='2023년 12월(예측) 상승률',
        marker_color='#ffff00',
        offsetgroup=0,
        text=[f'{rate:.2f}' for rate in df['202312_predict']/df['202212_real']],
        textposition='outside'
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        x=df['AptName_EArea2'],
        y=df['202312_real']/df['202212_real'],
        name='2023년 12월(실제) 상승률',
        marker_color='#ff0000',
        offsetgroup=1,
        text=[f'{rate:.2f}' for rate in df['202312_real']/df['202212_real']],
        textposition='outside'
    ), row=1, col=2)

    # Update layout for better appearance
    fig.update_layout(
        title={
            'text': '아파트별 가격 예측 결과',
            'x': 0.5,  # Center-align the title
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='',
        yaxis_title='전용면적당 가격(만원)',
        yaxis2_title='상승률',
        barmode='group',
        bargap=0.15,  # Gap between bars of adjacent location coordinates
        bargroupgap=0.1,  # Gap between bars of the same location coordinates
        legend_title_text='날짜 및 상승률',
        template='plotly_white'
    )

    # Adjust the subplot titles and layout
    fig.update_annotations(font_size=12)
    fig.update_layout(
        xaxis1=dict(title=''),
        yaxis1=dict(title='전용면적당 가격(만원)'),
        xaxis2=dict(title=''),
        yaxis2=dict(title='상승률')
    )

    return fig