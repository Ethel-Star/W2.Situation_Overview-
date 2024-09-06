import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class DatabaseConnector:
    def __init__(self):
        self.dbname = os.getenv('DB_NAME')
        self.user = os.getenv('DB_USER')
        self.password = os.getenv('DB_PASSWORD')
        self.host = os.getenv('DB_HOST')
        self.port = os.getenv('DB_PORT')
        self.connection = None

    def connect(self):
        try:
            self.connection = psycopg2.connect(
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
            print("Connection successful.")
        except Exception as e:
            print(f"An error occurred while connecting to the database: {e}")

    def execute_query(self, query, params=None):
        if self.connection is None:
            print("No connection to the database.")
            return None

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                results = cursor.fetchall()
                colnames = [desc[0] for desc in cursor.description]
                df = pd.DataFrame(results, columns=colnames)
                return df
        except Exception as e:
            print(f"An error occurred while executing the query: {e}")
            return None

    def execute_update(self, query, params=None):
        """For INSERT, UPDATE, DELETE queries."""
        if self.connection is None:
            print("No connection to the database.")
            return

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                self.connection.commit()  # Commit changes to the database
                print("Query executed and committed successfully.")
        except Exception as e:
            print(f"An error occurred during update: {e}")

    def close_connection(self):
        if self.connection is not None:
            try:
                self.connection.close()
                print("Connection closed.")
            except Exception as e:
                print(f"An error occurred while closing the connection: {e}")
        else:
            print("No connection to close.")

