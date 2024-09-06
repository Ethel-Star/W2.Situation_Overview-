import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
        ax.setTitle(title)
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

    def plot_univariate_analysis(self, variables):
        """Create and display histograms for specified quantitative variables."""
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
        """Perform bivariate analysis between each application and total DL+UL data."""
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

        plt.figure(figsize=(10, 6))
        plt.bar(range(1, n_components + 1), explained_variance, alpha=0.6, color='b', label='Individual Explained Variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.title('PCA Explained Variance')
        plt.show()

        return pca_data, explained_variance
    