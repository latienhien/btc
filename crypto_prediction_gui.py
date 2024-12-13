import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QComboBox, QLabel, QPushButton, QDateEdit)
from PyQt5.QtCore import Qt, QDate
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import Holt
from PyQt5.QtGui import QPixmap

class CryptoPredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Cryto Prediction')
        self.setStyleSheet('''
            QMainWindow {
                background-color: #CCCCCC;
            }
            QLabel#logo {
                font-size: 20px;
                font-weight: bold;
                color: #4A4A4A;
            }
            QPushButton.menu_button {
                background-color: transparent;
                border: none;
                color: #4A4A4A;
                font-size: 16px;
                margin: 0 15px;
            }
            QPushButton.menu_button.active {
                color: #6A5ACD;
                font-weight: bold;
            }
            QPushButton#predict_btn {
                background-color: #6A5ACD;
                color: white;
                font-size: 16px;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
            }
            QPushButton#predict_btn:hover {
                background-color: #594DBD;
            }
            QComboBox, QDateEdit {
                padding: 8px;
                border: 1px solid #CCC;
                border-radius: 5px;
                font-size: 14px;
            }
            QLabel {
                font-size: 14px;
                color: #4A4A4A;
            }
        ''')
        
        # Container chính
        # Main container
        container = QWidget()
        self.setCentralWidget(container)
        main_layout = QVBoxLayout(container)
        
        # Header
        header = QWidget()
        header_layout = QHBoxLayout(header)
        
        # Logo và crypto selection
        logo_container = QWidget()
        logo_layout = QHBoxLayout(logo_container)

        # Logo image
        logo_img = QLabel()
        logo_pixmap = QPixmap(r"C:\Users\MY PC\Downloads\dataset\Bitcoin-Logo.png")
        logo_pixmap = logo_pixmap.scaled(50, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo_img.setPixmap(logo_pixmap)

        # Logo text
        logo_text = QLabel("Nhóm 2")
        logo_text.setObjectName("logo")

        logo_layout.addWidget(logo_img)
        logo_layout.addWidget(logo_text)
        logo_layout.setSpacing(10)  # Khoảng cách giữa logo và text
        header_layout.addWidget(logo_container, alignment=Qt.AlignLeft)
        
        
        # Phần chọn tiền ảo
        crypto_container = QWidget()
        crypto_layout = QHBoxLayout(crypto_container)
        self.crypto_combo = QComboBox()
        self.crypto_combo.addItems(['BTC', 'BNB', 'ETH', 'SOL', 'XRP'])
        crypto_layout.addWidget(self.crypto_combo)
        header_layout.addWidget(crypto_container)
        
        # Menu
        menu_widget = QWidget()
        menu_layout = QHBoxLayout(menu_widget)
        menu_items = ["Trang chủ", "Cộng đồng", "Thị trường", "Tin tức"]
        for item in menu_items:
            btn = QPushButton(item)
            btn.setProperty("class", "menu_button")
            if item == "Thị trường":
                btn.setProperty("class", "menu_button active")
            menu_layout.addWidget(btn)
        
        header_layout.addWidget(menu_widget, alignment=Qt.AlignRight)
        main_layout.addWidget(header)
        
        # Chart container
        chart_container = QWidget()
        chart_layout = QVBoxLayout(chart_container)
        self.figure = plt.figure(figsize=(10, 5))
        self.canvas = FigureCanvas(self.figure)
        chart_layout.addWidget(self.canvas)
        main_layout.addWidget(chart_container)
        
        # Filters
        filters = QWidget()
        filters_layout = QHBoxLayout(filters)
        
        # Date selection
        date_widget = QWidget()
        date_layout = QHBoxLayout(date_widget)
        start_label = QLabel('Ngày bắt đầu:')
        self.start_date = QDateEdit()
        self.start_date.setDate(QDate(2019, 1, 1))
        end_label = QLabel('Ngày kết thúc:')
        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        
        date_layout.addWidget(start_label)
        date_layout.addWidget(self.start_date)
        date_layout.addWidget(end_label)
        date_layout.addWidget(self.end_date)
        
        # Model selection
        model_widget = QWidget()
        model_layout = QHBoxLayout(model_widget)
        self.model_combo = QComboBox()
        self.model_combo.addItems(['Moving Average', 
                                  'Exponential Smoothing (α=0.1)', 
                                  'Exponential Smoothing (α tối ưu)',
                                  'Holt (tiêu chuẩn)',
                                  'Holt (tối ưu)',
                                  'Holt-Winters (tiêu chuẩn)',
                                  'Holt-Winters (tối ưu)'])
        model_layout.addWidget(self.model_combo)
        
        # Predict button
        self.predict_btn = QPushButton('Dự đoán')
        self.predict_btn.setObjectName("predict_btn")
        self.predict_btn.clicked.connect(self.make_prediction)
        
        filters_layout.addWidget(date_widget)
        filters_layout.addWidget(model_widget)
        filters_layout.addWidget(self.predict_btn)
        
        main_layout.addWidget(filters)
        
        self.setMinimumSize(800, 600)
        
        # Thêm connect cho crypto_combo
        self.crypto_combo.currentIndexChanged.connect(self.update_default_chart)
        
        # Hiển thị biểu đồ mặc định khi khởi động
        self.update_default_chart()
        
    def update_default_chart(self):
        """Hiển thị biểu đồ giá của coin được chọn"""
        crypto = self.crypto_combo.currentText()
        
        # Đọc dữ liệu
        df = pd.read_csv(f'{crypto}_5_years.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        
        # Vẽ biểu đồ
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Vẽ giá đóng cửa
        ax.plot(df.index, df['Close'], label='Giá thực tế', color='blue')
        
        # Thêm MA30 và MA60 để tham khảo
        ma30 = df['Close'].rolling(window=30).mean()
        ma60 = df['Close'].rolling(window=60).mean()
        ax.plot(df.index, ma30, label='MA30', color='orange', alpha=0.7)
        ax.plot(df.index, ma60, label='MA60', color='green', alpha=0.7)
        
        ax.set_title(f'Biểu đồ giá {crypto}')
        ax.set_xlabel('Ngày')
        ax.set_ylabel('Giá')
        ax.legend()
        plt.xticks(rotation=45)
        self.figure.tight_layout()
        self.canvas.draw()
        
    def make_prediction(self):
        crypto = self.crypto_combo.currentText()
        start_date = self.start_date.date().toPyDate()
        end_date = self.end_date.date().toPyDate()
        model = self.model_combo.currentText()
        
        # Đọc dữ liệu
        df = pd.read_csv(f'{crypto}_5_years.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df = df[(df.index >= str(start_date)) & (df.index <= str(end_date))]
        
        close_prices = df['Close']
        
        if model == 'Moving Average':
            # Tính toán Moving Average
            ma30 = close_prices.rolling(window=30).mean()
            ma60 = close_prices.rolling(window=60).mean()
            naive_forecast = close_prices.shift(1)
            
            # Vẽ đồ thị
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(df.index, close_prices, label='Giá thực tế', color='blue')
            ax.plot(df.index, naive_forecast, label='Naive Forecast', color='red', linestyle='--')
            ax.plot(df.index, ma30, label='MA30', color='green')
            ax.plot(df.index, ma60, label='MA60', color='purple')
            ax.set_title(f'Phân tích giá {crypto} với Moving Averages')
            
        elif model == 'Exponential Smoothing (α=0.1)':
            # SES với alpha = 0.1
            ses_fixed = SimpleExpSmoothing(close_prices).fit(smoothing_level=0.1, optimized=False)
            ses_fixed_values = ses_fixed.fittedvalues
            
            # Vẽ đồ thị
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(df.index, close_prices, label='Giá thực tế', color='blue')
            ax.plot(df.index, ses_fixed_values, label='SES (α=0.1)', color='orange')
            ax.set_title(f'Phân tích giá {crypto} với SES (α=0.1)')
            
        elif model == 'Exponential Smoothing (α tối ưu)':
            # Tìm alpha tối ưu
            train_size = int(len(close_prices) * 0.8)
            train = close_prices[:train_size]
            test = close_prices[train_size:]
            
            best_alpha = None
            best_rmse = float('inf')
            alphas = np.linspace(0.01, 0.99, 99)
            
            for alpha in alphas:
                model_temp = SimpleExpSmoothing(train).fit(smoothing_level=alpha, optimized=False)
                forecast = model_temp.forecast(len(test))
                rmse = np.sqrt(mean_squared_error(test, forecast))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_alpha = alpha
            
            # Áp dụng alpha tối ưu
            ses_opt = SimpleExpSmoothing(close_prices).fit(smoothing_level=best_alpha, optimized=False)
            ses_opt_values = ses_opt.fittedvalues
            
            # Vẽ đồ thị
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(df.index, close_prices, label='Giá thực tế', color='blue')
            ax.plot(df.index, ses_opt_values, label=f'SES (α={best_alpha:.3f})', color='brown')
            ax.set_title(f'Phân tích giá {crypto} với SES (α tối ưu={best_alpha:.3f})')
        
        elif model == 'Holt (tiêu chuẩn)':
            # Chia dữ liệu thành train và test
            train_size = int(len(close_prices) * 0.8)
            train_data = close_prices[:train_size]
            test_data = close_prices[train_size:]
            train_dates = df.index[:train_size]
            test_dates = df.index[train_size:]
            
            # Holt tiêu chuẩn
            holt_model = Holt(train_data, exponential=False).fit()
            holt_forecast = holt_model.forecast(len(test_data))
            
            # Vẽ đồ thị
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(train_dates, train_data, label='Train Data', color='blue')
            ax.plot(test_dates, test_data, label='Test Data', color='orange')
            ax.plot(test_dates, holt_forecast, label='Holt Forecast', color='green')
            ax.set_title(f'Phân tích giá {crypto} với Holt tiêu chuẩn')
        
        elif model == 'Holt (tối ưu)':
            # Chia dữ liệu thành train và test
            train_size = int(len(close_prices) * 0.8)
            train_data = close_prices[:train_size]
            test_data = close_prices[train_size:]
            train_dates = df.index[:train_size]
            test_dates = df.index[train_size:]
            
            smoothing_level_options = np.arange(0.1, 1.0, 0.1)
            smoothing_slope_options = np.arange(0.1, 1.0, 0.1)
            
            best_rmse = float('inf')
            best_params = None
            best_forecast = None
            
            for alpha in smoothing_level_options:
                for beta in smoothing_slope_options:
                    try:
                        model_temp = Holt(train_data, exponential=False).fit(
                            smoothing_level=alpha, 
                            smoothing_slope=beta
                        )
                        forecast = model_temp.forecast(len(test_data))
                        rmse = np.sqrt(mean_squared_error(test_data, forecast))
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_params = (alpha, beta)
                            best_forecast = forecast
                    except:
                        continue
            
            # Vẽ đồ thị
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(train_dates, train_data, label='Train Data', color='blue')
            ax.plot(test_dates, test_data, label='Test Data', color='orange')
            ax.plot(test_dates, best_forecast, 
                    label=f'Holt Forecast (α={best_params[0]:.2f}, β={best_params[1]:.2f})', 
                    color='green')
            ax.set_title(f'Phân tích giá {crypto} với Holt tối ưu\n(α={best_params[0]:.2f}, β={best_params[1]:.2f})')
        
        elif model == 'Holt-Winters (tiêu chuẩn)':
            # Chia dữ liệu thành train và test
            train_size = int(len(close_prices) * 0.8)
            train_data = close_prices[:train_size]
            test_data = close_prices[train_size:]
            train_dates = df.index[:train_size]
            test_dates = df.index[train_size:]
            
            # Holt-Winters tiêu chuẩn
            hw_model = ExponentialSmoothing(
                train_data, 
                trend='add', 
                seasonal='add', 
                seasonal_periods=365
            ).fit()
            hw_forecast = hw_model.forecast(len(test_data))
            
            # Vẽ đồ thị
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(train_dates, train_data, label='Train Data', color='blue')
            ax.plot(test_dates, test_data, label='Test Data', color='orange')
            ax.plot(test_dates, hw_forecast, label='Holt-Winters Forecast', color='green')
            ax.set_title(f'Phân tích giá {crypto} với Holt-Winters tiêu chuẩn')
        
        elif model == 'Holt-Winters (tối ưu)':
            # Chia dữ liệu thành train và test
            train_size = int(len(close_prices) * 0.8)
            train_data = close_prices[:train_size]
            test_data = close_prices[train_size:]
            train_dates = df.index[:train_size]
            test_dates = df.index[train_size:]
            
            # Tham số để tối ưu
            trend_options = ['add', 'mul']
            seasonal_options = ['add', 'mul']
            seasonal_periods_options = [30, 90, 180, 365]
            
            best_rmse = float('inf')
            best_params = None
            best_forecast = None
            
            for trend in trend_options:
                for seasonal in seasonal_options:
                    for periods in seasonal_periods_options:
                        try:
                            model_temp = ExponentialSmoothing(
                                train_data,
                                trend=trend,
                                seasonal=seasonal,
                                seasonal_periods=periods
                            ).fit()
                            forecast = model_temp.forecast(len(test_data))
                            rmse = np.sqrt(mean_squared_error(test_data, forecast))
                            if rmse < best_rmse:
                                best_rmse = rmse
                                best_params = (trend, seasonal, periods)
                                best_forecast = forecast
                        except:
                            continue
            
            # Vẽ đồ thị
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(train_dates, train_data, label='Train Data', color='blue')
            ax.plot(test_dates, test_data, label='Test Data', color='orange')
            ax.plot(test_dates, best_forecast, 
                    label=  f'Holt-Winters Forecast', color='green')
            ax.set_title(f'Phân tích giá {crypto} với Holt-Winters tối ưu')
        
        ax.set_xlabel('Ngày')
        ax.set_ylabel('Giá')
        ax.legend()
        plt.xticks(rotation=45)
        self.figure.tight_layout()
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CryptoPredictionApp()
    window.show()
    sys.exit(app.exec_()) 
    
    