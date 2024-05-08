# Imports

# Data
import yfinance as yf

# Base
import numpy as np

# Preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Plotting
import matplotlib.pyplot as plt

# Metrics Plotting
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

# Classes

class TimeSeriesData:
    """
    This class creates a TimeSeriesData object using the yfinance library. This object downloads the data and allows some
    standard manipulations.  
    """

    def __init__(self, ticker_name, period='5y', interval='1d'):
        self.ticker_name = ticker_name
        self.period = period
        self.interval = interval
        self._Ticker = yf.Ticker(self.ticker_name)
        self.df = self._Ticker.history(
            period=self.period, interval=self.interval)

    def head(self):
        return self.df.head()

    def plot(self):
        # Credit to this article for the basic flow for this method https://medium.com/analytics-vidhya/visualizing-historical-stock-price-and-volume-from-scratch-46029b2c5ef9
        currency = self._Ticker.fast_info['currency']
        title = f'{self.ticker_name}'
        fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
        fig.tight_layout(pad=3)
        close = self.df['Close']
        vol = self.df['Volume']

        # Set the axis
        plot_price = axes[0]
        plot_vol = axes[1]

        # Plot the charts
        plot_price.plot(close, color='green')
        plot_vol.bar(vol.index, vol, width=15, color='darkgrey')
        # Move the ticks to the right
        plot_price.yaxis.tick_right()
        plot_vol.yaxis.tick_right()

        # Set the labels
        plot_price.set_ylabel(f'{currency}')
        plot_price.yaxis.set_label_position('right')
        plot_vol.set_ylabel('Volume')
        plot_vol.yaxis.set_label_position('right')

        # Set the grid lines
        plot_price.grid(axis='y', color='gainsboro',
                        linestyle='-', linewidth=0.5)
        plot_vol.grid(axis='y', color='gainsboro',
                      linestyle='-', linewidth=0.5)

        # Remove the top left borders
        for plot in [plot_price, plot_vol]:
            plot.spines['top'].set_visible(False)
            plot.spines['left'].set_visible(False)
            plot.spines['left'].set_color('grey')
            plot.spines['bottom'].set_color('grey')

        fig.suptitle(title)
        plt.show()

    def describe(self):
        return self.df.describe()


class FinancialTimeSeriesClassifier:
    """
    Classifier class that takes 2 arrays, X and y, and allows various manipulations using tools from Scikit Learn. 
    # Original reference Kannan Singaravelu, with some tweaks
    """
    def __init__(self, X, y, testsize=0.2):
        self.testsize = testsize
        self.X = X
        self.y = y
        # Split training and testing data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.testsize, random_state=0, shuffle=False)

    # Fit the training data and then predict y based on X_test. USe a pipeline for reproduction

    def fit_predict(self, estimator, transformer=None):

        try:
            # Use a pipeline to create the model
            model = Pipeline([('classifier', estimator)])
            if transformer:
                model = Pipeline([('scaler', transformer), ('classifier', estimator)])

            # Fit the model
            model.fit(self.X_train, self.y_train)

        except Exception as e:
            print(str(e))

        self.y_pred_test = model.predict(self.X_test)
        self.y_pred_train = model.predict(self.X_train)
        self.model = model

    def plot_confusion_matrix(self):
        confusion = ConfusionMatrixDisplay.from_estimator(
            self.model,
            self.X_test,
            self.y_test,
            display_labels=self.model.classes_,
            cmap='Dark2'
        )
        plt.title('Confusion matrix')
        plt.show()

    def plot_roc_curve(self):
        roc = RocCurveDisplay.from_estimator(
            self.model,
            self.X_test,
            self.y_test,
            name=self.model['classifier'],
            color='green'
        )
        plt.title("AUC-ROC Curve \n")
        plt.plot([0, 1], [0, 1], linestyle="--", label='Random 50:50', color='red')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
        plt.show()


# Functions

def create_features(time_series):
    # Original reference Kannan Singaravelu, with some tweaks
    df = time_series.df.copy()
    multiplier = 2
    # Features
    # Open close - % difference between closing and opening price on any particular day
    df['OC'] = df['Close'] / df['Open'] - 1
    # High low - % difference between high price and low price on the day
    df['HL'] = df['High'] / df['Low'] - 1
    # Gap - % difference between prior day close and current day open
    df['GAP'] = df['Open'] / df['Close'].shift(1) - 1
    # Log return - natural log of % change in closing
    df['RET'] = np.log(df['Close'] / df['Close'].shift(1))
    # Sigh of return
    df['SIGN'] = np.where(df['RET'] < 0, -1, 1)
    # Create features for different time periods (numbers represent days)
    periods = [7, 14, 28]
    for i in periods:
        # % change in closing price
        df['PCHG' + str(i)] = df['Close'].pct_change(i)
        # % change in volume traded
        df['VCHG' + str(i)] = df['Volume'].pct_change(i)
        # Sum of log return over period
        df['RET' + str(i)] = df['RET'].rolling(i).sum()
        # Price moving average
        df['MA' + str(i)] = df['Close'].rolling(i).mean()
        # Price EMA
        df['EMA' + str(i)] = df['Close'].ewm(span=i, adjust=False).mean()
        # % change volume moving average
        df['VMA' + str(i)] = df['Volume'] / df['Volume'].rolling(i).mean()
        # open close mean
        df['OC' + str(i)] = df['OC'].rolling(i).mean()
        # high low mean
        df['HL' + str(i)] = df['HL'].rolling(i).mean()
        # Gap mean
        df['GAP' + str(i)] = df['GAP'].rolling(i).mean()
        # Standard deviation of log returns over period
        df['STD' + str(i)] = df['RET'].rolling(i).std()
        # Upper bollinger band mean +- standard deviation x multiplier
        df['UB' + str(i)] = df['Close'].rolling(i).mean() + \
            df['Close'].rolling(i).std() * multiplier
        # Lower bollinger band
        df['LB' + str(i)] = df['Close'].rolling(i).mean() - \
            df['Close'].rolling(i).std() * multiplier
        # Momentum
        df['MOM' + str(i)] = df['Close'] - df['Close'].shift(i)
    
    # Reorder the columns into alphabetical order for easier analysis and visualization
    new_column_order = df.columns.sort_values()
    df = df[new_column_order]
    # Drop NaN values and other features that we won't use
    df.dropna(inplace=True)

    return df

def create_dependent_variable(X, threshold=0.0025):
    df = X.copy()
    df['Label'] = np.where(df['RET'].shift(-1) > threshold, 1, 0)
    return df['Label']
