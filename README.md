# Telecom User Analysis and Engagement Project

## Project Overview

This project focuses on analyzing telecom customer behavior using xDR (data session) records. It aims to provide insights into handset usage, customer engagement, and data consumption patterns across various applications. The insights derived from this analysis will help telecom companies enhance their marketing strategies, optimize network resources, and improve overall customer satisfaction.

## Objectives

1. **User Overview Analysis**
   - Identify the top handsets used by customers and their manufacturers.
   - Analyze customer behavior across various applications such as Social Media, Google, Email, YouTube, Netflix, Gaming, and Others.
   - Perform Exploratory Data Analysis (EDA) to uncover patterns, trends, and insights. Handle missing values and outliers using quantitative methods to improve data quality.

2. **User Engagement Analysis**
   - Track customer engagement metrics, including session frequency, session duration, and data usage.
   - Use k-means clustering to segment users into different engagement levels, aiding in network resource allocation and marketing efforts.

## Project Structure

```bash
project-directory/
│
├── data/
│   └── postegressql/              # Database containing the raw telecom dataset
│
├── notebooks/
│   ├── user_overview_analysis.ipynb    # Jupyter Notebook for Task 1 analysis
│   ├── user_engagement_analysis.ipynb  # Jupyter Notebook for Task 2 analysis
│
├── src/
│   ├── utils/                      # Folder containing utility scripts
│   │   ├── __init__.py             # Initialization script for the utils package
│   │   
├── scripts/
│   ├── connection.py                # PostgreSQL database connection script
│
└── README.md                        # Project documentation and instructions
