import sqlite3

def create_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create a table for storing cheque data
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS cheque_data (
        id TEXT PRIMARY KEY,
        client TEXT,
        amount TEXT,
        date TEXT,
        corrected_text TEXT
    )
    ''')

    conn.commit()
    conn.close()

# Replace 'path/to/your/database.db' with your desired path
create_database('/Users/mac/ChequeFixerATB/mlapp/database.db')
