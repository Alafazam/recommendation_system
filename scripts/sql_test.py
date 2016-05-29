import sqlite3
conn = sqlite3.connect('MovieLens.db')
c = conn.cursor()
c.execute("SELECT * FROM user_similarity_matrix WHERE userID=1 AND otherID!=1 ORDER BY similarity DESC LIMIT 100;")
print c.fetchall()
