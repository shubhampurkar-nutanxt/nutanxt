import sqlite3
import os
from datetime import datetime
import pymysql
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import requests
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
import base64
import json
from cryptography.hazmat.backends import default_backend

try:
    with open('/home/pi/OakRaman2/assets/scanner_config/app_settings.json', 'r') as f:
        app_settings = json.load(f)
        device_name = app_settings.get('machine_id', '000')  # Using .get() to handle missing key
        f.close()
except Exception as e:
    print("Error loading app settings:", e)
    device_name = '000'  # Default value in case of an error

print("Device_Name: ",device_name)


DB_PATH = "/home/pi/OakRaman2/oakapp/drugs_assets/ID2.db"

PRIVATE_KEY_PATH = '/home/pi/OakRaman2/auth/private_key.pem'

def load_private_key():
    with open(PRIVATE_KEY_PATH, 'rb') as f:
        return serialization.load_pem_private_key(f.read(), password=None,backend=default_backend())

def create_database():
    if not os.path.exists(DB_PATH):
        print(f"Creating database: {DB_PATH}")
    else:
        print(f"Database already exists: {DB_PATH}")
    
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

   
    create_table_query = """
    CREATE TABLE IF NOT EXISTS ID2_Reference (
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        Scan_id TEXT NOT NULL UNIQUE,
        xcal BLOB,
        feedback TEXT NOT NULL,
        Date_Time DATETIME DEFAULT (datetime('now', 'localtime'))
    );
    """
    try:
        cursor.execute(create_table_query)
        conn.commit()
    
    except sqlite3.Error as e:
        print(f"Error creating table: {e}")
    finally:
        conn.close()


def fetch_latest_record():
   
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

   
    select_query = """
    SELECT Date_Time FROM ID2_Reference
    ORDER BY Date_Time DESC LIMIT 1;
    """

    try:
        cursor.execute(select_query)
        latest_record = cursor.fetchone()
        if latest_record:
            return latest_record[0]
        else:
            print("No records found.")
            return None
    except sqlite3.Error as e:
        print(f"Error fetching latest record: {e}")
    finally:
        conn.close()

def insert_data_one_by_one(data_list):
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    insert_query = """
    INSERT INTO ID2_Reference (Scan_id, xcal, feedback, Date_Time)
    VALUES (?, ?, ?, ?)
    """

    try:
       
        for item in data_list:
            cursor.execute(insert_query, (item['scan_id'], f"{item['xcal']}", item['feedback'], item['Date_Time']))
            conn.commit()  
        print(f"Inserted {len(data_list)} records successfully.")
    except sqlite3.Error as e:
        print(f"Error inserting data: {e}")
    finally:
        conn.close()





def main():
    create_database()

    last_date = fetch_latest_record()
    if not last_date:
        last_date_str = '2024-06-12 02:02:37'
    else:
        last_date_str = last_date


    message = json.dumps({"date": last_date_str})
    with open(PRIVATE_KEY_PATH, "rb") as f:
        private_key = serialization.load_pem_private_key(f.read(), password=None,backend=default_backend())

    signature = private_key.sign(
        message.encode(),
        padding.PKCS1v15(),
        hashes.SHA256()
    )
    signature_b64 = base64.b64encode(signature).decode()

    payload = {
        "device_name": device_name,
        "message": message,
        "signature": signature_b64
    }

    try:
        response = requests.post("https://customer.narcranger.com/api/id2_data", json=payload)

        if response.status_code == 200:
            res_data = response.json()
            if res_data.get("success") and isinstance(res_data.get("data"), list):
                insert_data_one_by_one(res_data["data"])
            else:
                print("API responded but no valid data to insert.")
        else:
            print(f"API error: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"Failed to fetch data from API: {e}")

main()
