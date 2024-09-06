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
    def compute_and_plot_cluster_metrics(self, metrics):
        """Compute and plot cluster metrics."""
        # Group by the cluster and compute the metrics
        cluster_metrics = self.df.groupby('Cluster')[metrics].agg(['min', 'max', 'mean', 'sum'])
        cluster_metrics = cluster_metrics.reset_index()
        print(cluster_metrics)

        # Plotting function
        def plot_metrics(metrics_df, metrics, metric_names):
            n_metrics = len(metrics)
            fig, axes = plt.subplots(n_metrics, 4, figsize=(20, n_metrics * 5), sharex=True)
            fig.suptitle('Cluster Metrics Analysis')

            for i, metric in enumerate(metrics):
                for j, stat in enumerate(['min', 'max', 'mean', 'sum']):
                    ax = axes[i, j]
                    ax.bar(metrics_df['Cluster'], metrics_df[(metric, stat)], color='skyblue')
                    ax.set_title(f'{metric_names[i]} {stat.capitalize()}')
                    ax.set_xlabel('Cluster')
                    ax.set_ylabel('Value')
        
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

        metric_names = [metric.replace('_', ' ').title() for metric in metrics]
        # Plot metrics
        plot_metrics(cluster_metrics, metrics, metric_names)
    def aggregate_user_traffic(self):
        # Aggregating user total traffic per application using MSISDN
        self.application_traffic = self.df.groupby('MSISDN').agg({
            'Social Media DL (MB)': 'sum',
            'Google DL (MB)': 'sum',
            'Netflix DL (MB)': 'sum',
        }).reset_index()
        print("Aggregated Data (first 5 rows):\n", self.application_traffic.head())  # Debug

    def top_engaged_users(self, top_n=10):
        # Convert MSISDN to string to handle large numbers
        self.application_traffic['MSISDN'] = self.application_traffic['MSISDN'].astype(str)

        # Derive the top 10 most engaged users (MSISDN) per application
        self.top_10_users_per_app = {}
        for app in ['Social Media DL (MB)', 'Google DL (MB)', 'Netflix DL (MB)']:
            self.top_10_users_per_app[app] = self.application_traffic.nlargest(top_n, app)
            print(f"Top 10 users for {app} (first 5 rows):\n", self.top_10_users_per_app[app].head())  # Debug

    def plot_top_applications(self):
        # Plotting top 3 most used applications
        top_apps = ['Social Media DL (MB)', 'Google DL (MB)', 'Netflix DL (MB)']
        fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)

        for i, app in enumerate(top_apps):
            data = self.top_10_users_per_app[app]
            print(f"Plotting data for {app}:\n", data)  # Debug

            if len(data) > 0:  # Ensure there are users to plot
                axes[i].bar(data['MSISDN'], data[app], color='skyblue')
                axes[i].set_title(app)
                axes[i].set_xlabel('MSISDN')
                axes[i].set_ylabel('Total Traffic (MB)')
                axes[i].tick_params(axis='x', rotation=90)
            else:
                axes[i].set_title(f"No Data for {app}")

        plt.tight_layout()
        plt.show()
    def cluster_analysis(self, columns_to_scale, max_k=10, n_clusters=None, use_mini_batch=False):
    
        # Preprocess the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df[columns_to_scale])

        # Optionally apply PCA for dimensionality reduction
        pca = PCA(n_components=min(len(columns_to_scale), 2))
        reduced_data = pca.fit_transform(scaled_data)

        if n_clusters is None:
            # Find optimal k
            k_values = range(1, max_k + 1)
            inertia = []

            for k in k_values:
                kmeans = MiniBatchKMeans(n_clusters=k, random_state=42) if use_mini_batch else KMeans(n_clusters=k, random_state=42)
                kmeans.fit(reduced_data)
                inertia.append(kmeans.inertia_)

            plt.figure(figsize=(8, 6))
            plt.plot(k_values, inertia, marker='o')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Inertia')
            plt.title('Elbow Method for Optimal k')
            plt.grid(True)
            plt.show()

            # Determine the optimal k based on the elbow method
            optimal_k = int(input("Enter the optimal number of clusters (k): "))
        else:
            optimal_k = n_clusters

        # Apply k-means or MiniBatchKMeans clustering
        kmeans = MiniBatchKMeans(n_clusters=optimal_k, random_state=42) if use_mini_batch else KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(reduced_data)

        # Add cluster labels to the DataFrame
        self.df['Cluster'] = clusters

        # Summarize cluster statistics
        cluster_summary = self.df.groupby('Cluster')[columns_to_scale].mean()

        return self.df, cluster_summary, optimal_k

    def plot_univariate_analysis(self, variables):
        num_vars = len(variables)
        num_cols = 4
        num_rows = int(np.ceil(num_vars / num_cols))  # Calculate number of rows needed

        colors = sns.color_palette("husl", num_vars)  # Generate a color palette with distinct colors

        # Create histograms
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5), squeeze=False)
        axes = axes.flatten()  # Flatten axes for easy iteration
        
        for i, var in enumerate(variables):
            if var in self.df.columns:
                sns.histplot(self.df[var].dropna(), kde=False, ax=axes[i], color=colors[i])
                axes[i].set_title(f'{var} - Histogram')
                axes[i].set_xlabel(var)
                axes[i].set_ylabel('Frequency')
            else:
                axes[i].set_title(f'{var} - Column Not Found')
                axes[i].text(0.5, 0.5, 'Column Not Found', ha='center', va='center')
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    def bivariate_analysis(self, app_vars, total_dl_col, total_ul_col):
        
        results = {}
        
        for app_var in app_vars:
            if app_var in self.df.columns:
                # Compute Pearson correlation
                correlation_dl = self.df[[app_var, total_dl_col]].dropna().corr().iloc[0, 1]
                correlation_ul = self.df[[app_var, total_ul_col]].dropna().corr().iloc[0, 1]

                results[app_var] = {
                    'Correlation with Total DL (MB)': correlation_dl,
                    'Correlation with Total UL (MB)': correlation_ul
                }

                # Scatter plot with regression line
                plt.figure(figsize=(14, 6))

                plt.subplot(1, 2, 1)
                sns.regplot(x=app_var, y=total_dl_col, data=self.df, scatter_kws={'s':10}, line_kws={'color':'red'})
                plt.title(f'{app_var} vs {total_dl_col}')
                plt.xlabel(app_var)
                plt.ylabel(total_dl_col)

                plt.subplot(1, 2, 2)
                sns.regplot(x=app_var, y=total_ul_col, data=self.df, scatter_kws={'s':10}, line_kws={'color':'blue'})
                plt.title(f'{app_var} vs {total_ul_col}')
                plt.xlabel(app_var)
                plt.ylabel(total_ul_col)

                plt.tight_layout()
                plt.show()

        return pd.DataFrame(results).T

    def compute_correlation_matrix(df, variables):
        correlation_matrix = df[variables].corr()
        print("Correlation Matrix:\n", correlation_matrix)
    
        # Plot the correlation matrix as a heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.show()

    def perform_pca(self, n_components=2, columns=None):
        """Perform PCA to reduce dimensionality of the dataset."""
        if columns is not None:
            numeric_df = self.df[columns].select_dtypes(include=[np.number])
        else:
            numeric_df = self.df.select_dtypes(include=[np.number])

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)

        pca = PCA(n_components=n_components)
        pca_data = pca.fit_transform(scaled_data)

        explained_variance = pca.explained_variance_ratio_

        return pca_data, explained_variance
    