"""
db_service.py
    Die Klasse stellt Methoden zum Abfragen von Daten aus einer Datenbank und zur Authentifizierung von Benutzern bereit.
    _Hinweis_ Der gesamte Code dieser Datei wurde von Simon Krissel, 
    einer studentischen Hilfskraft an der Hochschule RheinMain,
    geschrieben und nicht von mir, Lukas Bröning. 
    Folgende Veränderungen wurden am Original vorgenommen, um besser damit arbeiten zu können:
        - Umbenennen der Klasse und des Dateinamens
        - Einfügen von Kommentaren
        - Entfernen englischer Kommentare
        - Umbenennen und Anpassung der authenticate Methode in check_User
        - Entfernen der Variablen label und timestamp in SQL GET-Abfrage und columns Variable
        - Auskommentieren von irrelevanten Prints

        PyMySQL Version 1.0.2
        pandas Version 1.3.1
"""
import pymysql
import pandas as pd


class Database_Service(object):

    # Konstruktor, initialisiert eine Verbindung zur Datenbank
    def __init__(self):
        self.mydb = pymysql.connect(
            host="localhost",
            user="your_user",
            password="your_password",
            database="your_database"
        )
        
        # Aktiviert den Autocommit-Modus, um neue Daten sofort verfügbar zu machen
        self.mydb.autocommit(True)

        # Erstellt einen Cursor, um Abfragen an die Datenbank zu senden
        self.mycursor = self.mydb.cursor()


    # Methode zum Abfragen von Daten für einen bestimmten Benutzer
    def get(self, user, limit=140):
        # SQL-Abfrage definieren, um Daten aus der Tabelle 'datensammlung' abzurufen
        sql = "SELECT accelUserX, accelUserY, accelUserZ, attitudePitch, attitudeRoll, attitudeYaw, gravityX, gravityY, gravityZ, gyroX, gyroY, gyroZ FROM datensammlung WHERE user_id=%s ORDER BY timestamp DESC LIMIT %s"

        # Überprüfung, ob eine Verbindung zur Datenbank besteht
        if not self.mydb.open:
            print("WARNING: SQL-DB disconnected! Reconnecting...")
            self.mysql.ping(reconnect=True)
        
        # SQL-Abfrage ausführen, um Daten abzurufen
        self.mycursor.execute(sql, (user, limit))

        # Abrufen aller Zeilen, die von der Abfrage zurückgegeben werden, und Speichern in einer Liste
        from_db = []
        for result in self.mycursor.fetchall():
            result = list(result)
            from_db.append(result)

        # Spaltennamen für den Pandas-DataFrame festlegen
        columns = ['accelUserX', 'accelUserY', 'accelUserZ', 'attitudePitch', 'attitudeRoll', 'attitudeYaw', 'gravityX', 'gravityY', 'gravityZ', 'gyroX', 'gyroY', 'gyroZ']
        # print(from_db[-1])
        # print(from_db)

        # DataFrame erstellen und zurückgeben
        return pd.DataFrame(from_db, columns=columns)


    # Methode zur Überprüfung der Verfügbarkeit des Benutzers
    def check_user(self, user):
        # SQL-Abfrage definieren, um Benutzerdaten abzurufen und zu überprüfen
        sql = "SELECT COUNT(1) FROM users WHERE user_id=%s"
        
        # Überprüfung, ob eine Verbindung zur Datenbank besteht
        if not self.mydb.open:
            print("WARNING: SQL-DB disconnected! Reconnecting...")
            self.mysql.ping(reconnect=True)
        
        # SQL-Abfrage ausführen und true/false zurückgeben
        self.mycursor.execute(sql, (user))
        return self.mycursor.fetchone()[0] > 0


    # Methode zur Überprüfung der Authorisierung des Benutzers
    def authenticate(self, user, auth):
        # SQL-Abfrage definieren, um Benutzerautorisierung zu prüfen
        sql = "SELECT COUNT(1) FROM users WHERE user_id=%s AND authorization=%s;"

        # Überprüfung, ob eine Verbindung zur Datenbank besteht
        if not self.mydb.open:
            print("WARNING: SQL-DB disconnected! Reconnecting...")
            self.mysql.ping(reconnect=True)

        # SQL-Abfrage ausführen und true/false zurückgeben
        self.mycursor.execute(sql, (user, auth))
        return self.mycursor.fetchone()[0] > 0


    # Methode zum Schließen der Verbindung zur Datenbank und des Cursors
    def close(self):
        self.mycursor.close()
        self.mydb.close()
