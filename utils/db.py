import os
import sqlite3

DATABASE_FOLDER = "database"
DATABASE_PATH = os.path.join(DATABASE_FOLDER, "history.db")


def get_connection():
    os.makedirs(DATABASE_FOLDER, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    connection = get_connection()
    cursor = connection.cursor()

    # ===== РЕНТГЕН =====
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            result_class TEXT NOT NULL,
            normal_probability TEXT NOT NULL,
            viral_probability TEXT NOT NULL,
            bacterial_probability TEXT NOT NULL,
            covid_probability TEXT DEFAULT '0%',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("PRAGMA table_info(analysis_history)")
    columns = [column[1] for column in cursor.fetchall()]

    if "covid_probability" not in columns:
        cursor.execute("""
            ALTER TABLE analysis_history
            ADD COLUMN covid_probability TEXT DEFAULT '0%'
        """)

    # ===== МРТ МОЗГА =====
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS brain_analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            result_class TEXT NOT NULL,
            glioma_probability TEXT NOT NULL,
            meningioma_probability TEXT NOT NULL,
            pituitary_probability TEXT NOT NULL,
            normal_probability TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    connection.close()


# ===== СОХРАНЕНИЕ РЕНТГЕНА =====
def save_analysis(filename, result_class, normal_probability, viral_probability, bacterial_probability, covid_probability):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO analysis_history (
            filename,
            result_class,
            normal_probability,
            viral_probability,
            bacterial_probability,
            covid_probability
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        filename,
        result_class,
        normal_probability,
        viral_probability,
        bacterial_probability,
        covid_probability
    ))

    connection.commit()
    connection.close()


# ===== СОХРАНЕНИЕ МРТ =====
def save_brain_analysis(filename, result_class, glioma_probability, meningioma_probability, pituitary_probability, normal_probability):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO brain_analysis_history (
            filename,
            result_class,
            glioma_probability,
            meningioma_probability,
            pituitary_probability,
            normal_probability
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        filename,
        result_class,
        glioma_probability,
        meningioma_probability,
        pituitary_probability,
        normal_probability
    ))

    connection.commit()
    connection.close()


# ===== ОБЪЕДИНЕННАЯ ИСТОРИЯ =====
def get_unified_history():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT 
            id,
            filename,
            'Рентген' AS analysis_type,
            result_class,
            normal_probability,
            viral_probability,
            bacterial_probability,
            covid_probability,
            NULL AS glioma_probability,
            NULL AS meningioma_probability,
            NULL AS pituitary_probability,
            created_at
        FROM analysis_history

        UNION ALL

        SELECT 
            id,
            filename,
            'МРТ мозга' AS analysis_type,
            result_class,
            normal_probability,
            NULL AS viral_probability,
            NULL AS bacterial_probability,
            NULL AS covid_probability,
            glioma_probability,
            meningioma_probability,
            pituitary_probability,
            created_at
        FROM brain_analysis_history

        ORDER BY created_at DESC
    """)

    rows = cursor.fetchall()
    connection.close()
    return rows


# ===== ОЧИСТКА =====
def clear_analysis_history():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("DELETE FROM analysis_history")
    connection.commit()
    connection.close()


def clear_brain_analysis_history():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("DELETE FROM brain_analysis_history")

    connection.commit()
    connection.close()


def clear_unified_history():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("DELETE FROM analysis_history")
    cursor.execute("DELETE FROM brain_analysis_history")

    connection.commit()
    connection.close()