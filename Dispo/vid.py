from sqlalchemy import create_engine, text

engine = create_engine("mysql+pymysql://AdminNidec:u6Ehe987XBSXxa4@141.94.31.144:3306/indicator")

with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'indicator'
        AND table_name LIKE 'dispo_pdc_n%'
    """))
    tables = [row[0] for row in result.fetchall()]
    for table in tables:
        conn.execute(text(f"TRUNCATE TABLE indicator.{table}"))
        print(f"✅ Table vidée : indicator.{table}")