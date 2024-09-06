import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class DataUtils:
    def __init__(self, df):
        self.df = df

    def check_missing_values(self):
        """Checks and summarizes missing values in the DataFrame."""
        missing_summary = self.df.isnull().sum()
        missing_percentage = (missing_summary / len(self.df)) * 100
        missing_table = pd.DataFrame({
            'Missing Values': missing_summary,
            'Percentage': missing_percentage.map("{:.1f}%".format),
            'Dtype': self.df.dtypes
        })
        missing_table = missing_table[missing_table['Missing Values'] > 0]
        missing_table = missing_table.sort_values(by='Percentage', ascending=False)
        
        print(f"Total columns with missing values: {missing_table.shape[0]}")
        print(f"Top 5 columns with the most missing values:\n{missing_table.head()}")
        
        return missing_table

    def handle_missing_values(self):
        """Handles missing values by filling them or dropping columns with excessive missing values."""
        for col in self.df.columns:
            if self.df[col].dtype in ['int64', 'float64']:
                if self.df[col].skew() > 1:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                else:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
            elif self.df[col].dtype == 'object':
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        
        # Drop columns with more than 50% missing values
        threshold = 0.5 * len(self.df)
        high_missing_cols = self.df.isnull().sum()[self.df.isnull().sum() > threshold].index
        self.df = self.df.drop(columns=high_missing_cols)
        
        print("Missing values handled and high missing value columns dropped.")
        return self.df

    def detect_outliers(self, z_thresh=3):
        """Detects outliers based on Z-score threshold for numeric columns."""
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        outlier_dict = {}
        for column in numeric_columns:
            column_data = self.df[column]
            z_scores = np.abs(stats.zscore(column_data, nan_policy='omit'))
            z_scores = pd.Series(z_scores, index=self.df.index)
            outliers = self.df.index[z_scores > z_thresh]
            outlier_dict[column] = outliers

        print("Outlier detection complete.")
        return outlier_dict

    def fix_outliers(self, method='median', quantile_val=0.95):
        """Fix outliers in the DataFrame for numeric columns."""
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if method == 'median':
                self.df[column] = np.where(
                    self.df[column] > self.df[column].quantile(quantile_val), 
                    self.df[column].median(), 
                    self.df[column]
                )
            elif method == 'mean':
                self.df[column] = np.where(
                    self.df[column] > self.df[column].quantile(quantile_val), 
                    self.df[column].mean(), 
                    self.df[column]
                )
        return self.df

    def remove_outliers(self, z_thresh=3):
        """Remove outliers based on Z-score threshold for numeric columns."""
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            z_scores = np.abs(stats.zscore(self.df[column].dropna()))
            self.df[column] = np.where(z_scores > z_thresh, np.nan, self.df[column])
        
        self.df.dropna(inplace=True)
        
        print("Outliers removed.")
        return self.df

    def bytes_to_megabytes(self, bytes_value):
        """Converts bytes to megabytes."""
        return bytes_value / (1024 * 1024)

    def convert_bytes_to_megabytes(self):
        """Converts columns with 'Bytes' in their names to megabytes."""
        byte_columns = [col for col in self.df.columns if '(Bytes)' in col]
        
        for column in byte_columns:
            new_column_name = column.replace('(Bytes)', '(MB)')
            self.df[new_column_name] = self.df[column].apply(self.bytes_to_megabytes)
        
        print("Converted byte columns to megabytes.")
        return self.df
    
    def plot_bar_with_annotations(self, ax, data, labels, title, colors):
        bars = ax.bar(labels, data, color=colors)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f} MB', va='bottom', ha='center')

        ax.set_xlabel('Applications')
        ax.set_ylabel('Total Data (MB)')
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=45)
        
    def compute_statistics(self, quantitative_vars):
        """Compute and return basic statistical metrics for specified quantitative variables."""
        stats_summary = pd.DataFrame()

        for var in quantitative_vars:
            stats_summary[var] = {
                'Mean': self.df[var].mean(),
                'Median': self.df[var].median(),
                'Standard Deviation': self.df[var].std(),
                'Variance': self.df[var].var(),
                'Minimum': self.df[var].min(),
                'Maximum': self.df[var].max(),
                '25th Percentile': self.df[var].quantile(0.25),
                '75th Percentile': self.df[var].quantile(0.75)
            }
        
        return stats_summary

    def analyze_customer_engagement(self):
        """Analyze customer engagement metrics and return top 10 customers by various metrics."""
        self.df.rename(columns={
            'MSISDN/Number': 'MSISDN',
            'Dur. (ms)': 'Session_duration',
            'Total DL (Bytes)': 'DL_data',
            'Total UL (Bytes)': 'UL_data'
        }, inplace=True)

        # Group by MSISDN and aggregate the necessary columns
        user_engagement = self.df.groupby('MSISDN').agg({
            'MSISDN': 'count',                # Number of sessions (frequency)
            'Session_duration': 'sum',        # Total session duration
            'DL_data': 'sum',                 # Total download data
            'UL_data': 'sum'                  # Total upload data
        }).rename(columns={'MSISDN': 'xDR_sessions'}).reset_index()

        # Calculate the total data volume (DL + UL)
        user_engagement['Total_data'] = user_engagement['DL_data'] + user_engagement['UL_data']

        # Sort and get top 10 customers for each engagement metric
        top_10_sessions = user_engagement.sort_values(by='xDR_sessions', ascending=False).head(10)
        top_10_duration = user_engagement.sort_values(by='Session_duration', ascending=False).head(10)
        top_10_total_data = user_engagement.sort_values(by='Total_data', ascending=False).head(10)

        # Create a dictionary to store the results
        results = {
            'Top 10 Customers by Number of Sessions': top_10_sessions,
            'Top 10 Customers by Session Duration': top_10_duration,
            'Top 10 Customers by Total Data (Download + Upload)': top_10_total_data
        }

        return results
    def normalize_metrics(self):
        
        metrics = ['Session_duration', 'DL_data', 'UL_data', 'Total DL (MB)', 'Total UL (MB)']
        if not all(metric in self.df.columns for metric in metrics):
            missing_cols = [metric for metric in metrics if metric not in self.df.columns]
            raise ValueError("The DataFrame is missing the following columns: " + ", ".join(missing_cols))
        
        scaler = StandardScaler()
        self.df[metrics] = scaler.fit_transform(self.df[metrics])
        print("Metrics normalized.")
        return self.df
    
    def perform_kmeans_clustering(self, k=3):
        self.normalize_metrics()
        metrics = ['Session_duration', 'DL_data', 'UL_data', 'Total DL (MB)', 'Total UL (MB)']
        kmeans = KMeans(n_clusters=k, random_state=0)
        self.df['Cluster'] = kmeans.fit_predict(self.df[metrics])
        print("Cluster centers:\n", kmeans.cluster_centers_)
        cluster_counts = self.df['Cluster'].value_counts()
        print("Number of customers in each cluster:\n", cluster_counts)

        return self.df, kmeans
    def apply_pca(self, n_components=2):
        """Apply PCA to reduce the dimensionality of the data."""
        metrics = ['Session_duration', 'DL_data', 'UL_data', 'Total DL (MB)', 'Total UL (MB)']
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(self.df[metrics])
        if n_components == 2:
            self.df[['PC1', 'PC2']] = principal_components
        elif n_components == 3:
            self.df[['PC1', 'PC2', 'PC3']] = principal_components
        return self.df

    def plot_clusters(self):
        """Plot the clusters in 2D and 3D on subplots."""
        fig = plt.figure(figsize=(14, 7))

        # 2D plot
        ax1 = fig.add_subplot(121)
        scatter1 = ax1.scatter(self.df['Session_duration'], self.df['DL_data'], c=self.df['Cluster'], cmap='viridis')
        ax1.set_xlabel('Session Duration')
        ax1.set_ylabel('Download Data')
        ax1.set_title('Customer Clusters (2D)')
        fig.colorbar(scatter1, ax=ax1, label='Cluster')

        # 3D plot
        ax2 = fig.add_subplot(122, projection='3d')
        scatter2 = ax2.scatter(self.df['Session_duration'], self.df['DL_data'], self.df['UL_data'], c=self.df['Cluster'], cmap='viridis')
        ax2.set_xlabel('Session Duration')
        ax2.set_ylabel('Download Data')
        ax2.set_zlabel('Upload Data')
        ax2.set_title('Customer Clusters (3D)')
        fig.colorbar(scatter2, ax=ax2, label='Cluster')

        plt.tight_layout()
        plt.show()