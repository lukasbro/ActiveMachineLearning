'''
al_webserver.py
    Active Learning Webserver, Flask-Webanwendung mittels Template-Engine und HTTP-Methoden verwalten.
    Rendern des Active Learning Tools und des ML-Trainingstools.
    Modelltraining (ML-Tool) verwalten, Active Learning Prozess durchführen.

    Created by Lukas Broening in 2023.
    Tested for:
        macOS Monterey Version 12.6
        Python Version 3.11.2
        Terminal Version 2.12.7 (445)
        pandas Version 1.5.3
        numpy Version 1.23.3
        scikit-learn Version 1.2.1
        tsfel Version 0.1.5
        Jinja2 Version 3.1.2
        Flask Version 2.2.2
        modAL Version 0.4.1

        Linux-4.19.0-18-amd64-x86_64-with-debian-10.11
        Python Version 3.7.3
        Terminal Version 2.12.7 (445)
        pandas Version 1.3.1
        numpy Version 1.21.1
        scikit-learn Version 0.24.2
        tsfel Version 0.1.4
        joblib Version 1.0.1
        Jinja2 Version 3.1.2
        Flask Version 2.2.3
        modAL Version 0.4.1
'''

# Bibliotheken-Packages-Import
import os
import time
import joblib
import numpy as np
import pandas as pd

from db_service import Database_Service
from ml_trainingstool import ML_Tool
from multiprocessing import Manager

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling, classifier_uncertainty, entropy_sampling, classifier_entropy

from flask import Flask, render_template, send_from_directory, request, redirect, flash, jsonify, get_flashed_messages, Markup, url_for


# Flask-Anwendung initialisieren mit Pfad und Einstellungen für statische Dateien, geheime Schlüssel, Auto-Reload
app = Flask(__name__, static_url_path='/python3', static_folder='static')
app.config['SERVER_NAME'] = 'your-server.de'
app.config['SECRET_KEY'] = '00000000-0000-0000-0000-your-server-key'
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Verzeichnis-Pfade
data_dir = '/your_path/data/'
model_dir = '/your_path/trained_models/'

# Liste der verfügbaren Feature-Domänen, Query Strategien Algorithmen
feature_domain = ['statistical', 'temporal', 'spectral', 'all']
query_strategy = ['Uncertainty Sampling', 'Entropy Sampling']
algorithms = ['All', 'RandomForest', 'LogisticRegression', 'SVC', 'GaussianNB', 'DecisionTree', 'KNN',
              'MLP', 'AdaBoostClassifier', 'QuadraticDiscriminantAnalysis',
              'LinearDiscriminantAnalysis', 'SGDClassifier', 'GradientBoost']

# Globale Label-Antwort des Probanden und Statusdictionary für AL-Verfahren, Liste für Labels eines geladenen Datensatzes
label = None
unique_labels = []
status_al = {"status": "Vorbereitung", "finished": "", "error": "", "iteration": 0, "accuracy": 0.0, "uncertainty": 0.0}



# Flask-Route für die Hauptseite (index) der Anwendung
@app.route('/', methods=['GET'])
def index():
    '''
        Dient zum Rendern der Hauptseite der Anwendung mit den grundlegenden Informationen.
        Aufruf, wenn ein GET-Request auf der Route '/' empfangen wird.
                Returns:
                        Gerendertes HTML-Template, das Informationen zu CSV-Dateien, Labels, 
                        Feature Domain, Algorithmen, Benutzer, Labels pro Benutzer enthält.
    '''
    # Prüfe, ob Benutzer autorisiert ist, die Anwendung zu benutzen
    user = request.cookies.get("user_id")
    authorization = request.cookies.get("Authorization")
    db_service = Database_Service()
    if db_service.authenticate(user, authorization):

        # Liste der CSV-Dateinamen im Datenverzeichnis abrufen.
        csvnames = os.listdir(data_dir)
        csv_filenames = [f for f in csvnames if f.endswith('.csv')]

        # Laden erste CSV-Datei und Informationen
        filename_csv = os.path.join(data_dir, csv_filenames[0])
        unique_labels, unique_users, unique_labels_by_user = load_csv(
                filename_csv)

        # Liste von Tupeln für Feature Domain Auswahl erstellen
        domain_values = [(domain, domain == feature_domain[0])
                            for domain in feature_domain]
        
        db_service.close()
        return render_template('index.html', csv_filenames=csv_filenames,
                            unique_labels=unique_labels, selected_file=csv_filenames[0],
                            domain_values=domain_values, algorithms=algorithms,
                            unique_users=unique_users, unique_labels_by_user=unique_labels_by_user)
    else:
        db_service.close()
        return "<h1>Authorization Error</h1>"
    

    


# Flask-Route für die Active Learning Seite (active-ml) der Anwendung
@app.route("/activeml")
def activeml():
    '''
        Dient zum Rendern der AL-Seite der Anwendung mit den grundlegenden Informationen.
        Aufruf, wenn ein GET-Request auf der Route '/' empfangen wird.
                Returns:
                        Gerendertes HTML-Template, das Informationen zu Model-Dateien, Labels, 
                        Feature Domain, Window-Size, Daten und Score enthält.
    '''
    # Prüfe, ob Benutzer autorisiert ist, die Anwendung zu benutzen
    user = request.cookies.get("user_id")
    authorization = request.cookies.get("Authorization")
    db_service = Database_Service()
    if db_service.authenticate(user, authorization):

        # Liste von Modelldateinamen im Verzeichnes abrufen.
        modelnames = os.listdir(model_dir)
        model_filenames = [f for f in modelnames if f.endswith('.joblib')]

        # Lade erstes Model und Informationen
        model_filename = os.path.join(model_dir, model_filenames[0])
        training_information, ml_model, model_name, classes, data_saved, X_train, y_train, X_test, y_test = load_model(
            model_filename)
        model, score, time_created, window_size, domain, labels = training_information

        # Aktuellen Status initialisieren
        global status_al
        status_al.update({"status": "Vorbereitung", "error": "", "iteration": 0, "accuracy": 0, "uncertainty": 0})

        # Liste von Tupeln für Query Strategie Auswahl erstellen
        strategies = [(strategy, strategy == query_strategy[0])
                            for strategy in query_strategy]

        db_service.close()
        return render_template('active-ml.html',
                            model_filenames=model_filenames, selected_model=model_filenames[0],
                            model=model_name, score=score, window_size=window_size, domain=domain, labels=classes,
                            data_saved=data_saved, strategies=strategies)
    else:
        db_service.close()
        return "<h1>Authorization Error</h1>"


# Favicon zur Verfügung stellen
@app.route('/favicon.ico')
def favicon():
    '''
        Aufruf, wenn der Browser eine Anfrage zum Pfad "/favicon.ico" sendet. Verhindert Fehlermeldung in der Konsole.
                Returns:
                        Icon, das sich im Ordner "static" befindet aus demselben Verzeichnis wie die Hauptanwendung.
    '''
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')


# Informationen zu neu ausgewähltem Datensatz oder ML-Modell laden, Seite aktualisieren
@app.route('/handle_data', methods=['POST'])
def handle_data():
    '''
        Dient zum erneuten Rendern der Hauptseite oder AL-Seite der Anwendung mit den grundlegenden Informationen
        für einen neu ausgewählten CSV-Datensatz (index) oder ein neu ausgewähltes ML-Modell (active-ml).
        Aufruf, wenn ein POST-Request auf der Route 'handle_data' empfangen wird.
                Returns:
                        Gerendertes HTML-Template, das Informationen zum CSV-Datensatz oder zum ML-Modell enthält.
    '''
    selected_file = None
    selected_model = None

    # Wenn Auswahl einer neuen CSV-Datei (index)
    if request.form.get('selected_file'):
        # Namen der ausgewählten CSV-Datei abrufen und Datei laden
        selected_file = request.form['selected_file']
        filename = os.path.join(
            '/your_path/data/', selected_file)
        unique_labels, unique_users, unique_labels_by_user = load_csv(filename)

         # Liste der CSV-Dateinamen im Datenverzeichnis abrufen.
        csvnames = os.listdir(data_dir)
        csv_filenames = [f for f in csvnames if f.endswith('.csv')]

        # Liste von Tupeln für Feature Domain Auswahl erstellen
        domain_values = [(domain, domain == feature_domain[0])
                         for domain in feature_domain]

        # Rendern des Templates mit der aktualisierten Daten
        return render_template('index.html', csv_filenames=csv_filenames,
                               unique_labels=unique_labels, selected_file=selected_file,
                               domain_values=domain_values, algorithms=algorithms,
                               unique_users=unique_users, unique_labels_by_user=unique_labels_by_user)

    # Wenn Auswahl eines neuen ML-Models (active-ml)
    elif request.form.get('selected_model'):
        # Namen des ausgewählten Models abrufen und Model-Datei laden
        selected_model = request.form['selected_model']
        model_filename = os.path.join(model_dir, selected_model)
        training_information, ml_model, model_name, classes, data_saved, X_train, y_train, X_test, y_test = load_model(
            model_filename)
        model, score, time_created, window_size, domain, labels = training_information

        # Liste von Modelldateinamen im Verzeichnes abrufen.
        modelnames = os.listdir(model_dir)
        model_filenames = [f for f in modelnames if f.endswith('.joblib')]

        # Aktuellen Status initialisieren
        global status_al
        status_al.update({"status": "Vorbereitung", "error": "", "iteration": 0, "accuracy": 0, "uncertainty": 0})

        # Liste von Tupeln für Query Strategie Auswahl erstellen
        strategies = [(strategy, strategy == query_strategy[0])
                            for strategy in query_strategy]

        # Rendern des Templates mit der aktualisierten Daten
        return render_template('active-ml.html',
                               model_filenames=model_filenames, selected_model=selected_model,
                               model=model_name, score=score, window_size=window_size, domain=domain, labels=classes,
                               data_saved=data_saved, strategies=strategies)


# Define a function to load the CSV file and get unique labels
def load_csv(filename):

    global unique_labels

    df = pd.read_csv(filename)
    # print(df.columns)
    unique_labels = list(df['label'].unique())
    unique_labels.insert(0, 'All')

    if 'user_id' in df.columns:
        unique_users = list(df['user_id'].unique())
        unique_labels_by_user = list(df.groupby('user_id')['label'].unique())
    else:
        unique_users = ['None']
        unique_labels_by_user = list([df['label'].unique()])

    # print(unique_labels_by_user)
    # print(unique_labels)

    return unique_labels, unique_users, unique_labels_by_user


# Ausgwähltes ML-Modell laden, Informationen über das Modell zurückgeben (active-ml)
def load_model(filename):
    '''
        Lädt gespeichertes ML-Modell aus Joblib-Datei, gibt Informationen über das Modell und seine Parameter zurück.
                Parameter:
                        filename (String): Name der Joblib-Datei des Modells.
                Returns:
                        Tupel mit Informationen über das Modell und seine Parameter 
                        sowie das Modell selbst und die Trainings- und Testdaten.
    '''
    # Lade gespeichertes Modell
    loaded_object = joblib.load(filename)

    # Prüfe, ob das Objekt ein Tupel ist und somit Train-Test-Daten enthält
    if isinstance(loaded_object, tuple):
        model, X_train, y_train, X_test, y_test = loaded_object
        data_saved = 'vorhanden'
    else:
        model = loaded_object
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        data_saved = 'keine'

    # Namen des Modells abrufen
    model_name = type(model).__name__

    # Abrufen der Klassen-Label (wenn vorhanden)
    if hasattr(model, 'classes_'):
        classes = model.classes_
    else:
        classes = ""

    # Anzahl der Features abfragen (wenn vorhanden)
    if hasattr(model, 'n_features_'):
        n_features = model.n_features_
    else:
        n_features = ""

    # Anzahl der Klassen abfragen (wenn vorhanden)
    if hasattr(model, 'n_classes_'):
        n_classes = model.n_classes_
    else:
        n_classes = ""

    # Abrufen der Parameter des Modells
    # params = model.get_params()

    '''
    print("MODEL INFOS – Name: ", model_name,
          ", Anzahl Klassen: ", n_classes,
          ", Klassen: ", classes,
          ", Anzahl Features: ", n_features)
    '''

    # Aufteilen des Dateinamens in einzelne Modellinformationen
    filename_without_ext = os.path.splitext(filename)[0]
    filename_without_model_dir = filename_without_ext.replace(model_dir, '')
    training_information = filename_without_model_dir.split("_")
    # print(training_information)

    return training_information, model, model_name, classes, data_saved, X_train, y_train, X_test, y_test


# Ausgewähltes ML-Modell löschen (active-ml)
@app.route('/delete_model', methods=['POST'])
def delete_model():
    '''
        Löscht das ausgewählte Modell aus dem Model-Verzeichnis und lädt Active-ML-Seite neu.
        Aufruf, wenn ein POST-Request auf der Route '/delete_model' empfangen wird.
                Returns:
                        Aufruf von activeml(), um Seite zu aktualisieren.
    '''
    selected_model = request.form['delete_model']
    model_path = os.path.join(model_dir, selected_model)
    os.remove(model_path)
    return redirect('https://your-server.de/python3/activeml')


# Modelltraining des ML-Trainingstools (index)
@app.route('/ml_training', methods=['POST'])
def ml_training():
    '''
        Führt die Vorverarbeitung ausgewählter Daten, die Feature-Extraktion, Modelltraining und Klassifikation durch.
        Aufruf, wenn ein POST-Request auf der Route '/ml_training' empfangen wird.
        - load_filter_data(): lädt die Daten und filtert sie gemäß der Konfigurationen.
        - auto_feature_extraction(): extrahiert automatisch Features.
        - extract_labels(): gibt Label für die automatisch extrahierten Features zurück.
        - feature_selection(): führt eine Feature-Selektion durch, falls erforderlich.
        - classify(): trainiert ausgewählte Klassifikatoren, prognostiziert Testdaten, speichert Modell und Daten, gibt Ergebnisse zurück.
                Returns:
                        response (JSON): Ergebnisse des Trainings werden in einem JSON-Objekt zurückgegeben
    '''
    # Initialisiere ML_Tool-Instanz
    ml_tool = init_ml_training()

    # Messe Ausführungszeit
    start_time = time.time()

    # Lade Datensatz, filtere gemäß der Konfigurationen, trenne Sensordaten von Labels
    x_train, y_train, x_test, y_test = ml_tool.load_filter_data()

    # x_train, x_test, y_train, y_test = ml_tool.train_test_split(X, y, 0.8)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # Dataframes mit den häufigsten Labeln für jedes Segment generieren
    y_train = ml_tool.extract_labels(y_train)['label']
    y_test = ml_tool.extract_labels(y_test)['label']
    print(y_train.shape, y_test.shape, '\n \n')

    # Automatisches Extrahieren von Features
    x_train_feat = ml_tool.auto_feature_extraction(x_train)
    x_test_feat = ml_tool.auto_feature_extraction(x_test)
    print(x_train_feat.shape, x_test_feat.shape, '\n \n')

    # Feature Selektion (ml_tool prüft, ob JA/NEIN)
    X_train, X_test = ml_tool.feature_selection(
        x_train_feat, x_test_feat)
    print(X_train.shape, X_test.shape, '\n \n')

    # Aktuelle Zeit in Sekunden
    current_time = time.time()

    # Zeit in Struktur umwandeln
    time_struct = time.localtime(current_time)

    # Stunden, Minuten und Sekunden aus Struktur extrahieren
    hours = time_struct.tm_hour
    minutes = time_struct.tm_min
    seconds = time_struct.tm_sec

    # Aktuelle Zeit für Dateinamen
    current_time = "{:02d}{:02d}{:02d}".format(hours, minutes, seconds)

    # Trainiere Klassifikatoren, sage Testdaten voraus, speichere Modell und gebe Ergebnisse zurück
    classifiers_message, report_message, ml_filename = ml_tool.classify(
        X_train, X_test, y_train, y_test, current_time)

    # Aktuelle Zeit in lesbarem Format Stunden:Minuten:Sekunden
    current_time = "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)

    # Übermittle die Ergebnisse mittels Flash-Nachrichten, die im Template gerendert werden
    flash(f"<h2>Ergebnisse ({current_time})</h2>")
    flash("<hr>")
    flash("<h3>Ergebnisse - Algorithmen Ranking:</h3>")
    flash(classifiers_message)
    flash("<hr>")
    flash("<h3>Report des besten Algorithmus:</h3>")
    flash(report_message)
    flash(f'Modell gespeichert als: {ml_filename}.')

    # Messe Ausführungszeit und übermittle sie
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    flash("<hr>")
    flash("<h3>Ausführungszeit:</h3>")
    flash(f"{execution_time:.2f} sec")

    # Rufe alle geflashten Nachrichten ab
    messages = get_flashed_messages()
    # Erstelle und sende JSON-Response zurück
    response = jsonify({'messages': messages})
    response.headers.add('Content-Type', 'application/json')
    return response


# ML_Tool-Instanz mit Nutzer-Konfigurationen initialisieren
def init_ml_training():
    '''
        Verarbeitet die Eingabeformulare, extrahiert die Nutzerkonfigurationen,
        erstellt eine ML_Tool-Instanz mit den ausgewählten Konfigurationen.
                Returns:
                        ml_tool: ML_Tool-Instanz mit den ausgewählten Konfigurationen.
    '''

    global unique_labels

    # Lese alle Input-Felder nacheinander aus
    selected_file = request.form.get('selected_file')
    filepath = os.path.join(data_dir, selected_file)

    window_size = int(request.form.get('window_size'))

    hertz = int(request.form.get('hertz'))

    checked_labels = request.form.getlist('label')

    checked_selection = request.form.getlist('feature_selection')
    if checked_selection == []:
        selection = None
    else:
        selection = "checked"

    save_data = request.form.getlist('save_data')
    if save_data == []:
        save_data = None
    else:
        save_data = "checked"

    train_user_names = []
    test_user_names = []
    for name in request.form.keys():
        if name.startswith('train-checkbox-'):
            train_user_names.append(name)
        elif name.startswith('test-checkbox-'):
            test_user_names.append(name)

    # Entferne Präfix von jeder Eingabe - kann ['None'] sein!
    train_users = [name[len('train-checkbox-'):] for name in train_user_names]
    train_users = [int(user_id) for user_id in train_users]

    # kann ['None'] sein!
    test_users = []
    if len(test_user_names) == 0:
        print('No test checkboxes were checked')
    else:
        # Entferne Präfix von jeder Eingabe
        test_users = [name[len('test-checkbox-'):] for name in test_user_names]
        test_users = [int(user_id) for user_id in test_users]

    # Wenn "ALLE" bei Labels, Feature Domains oder Algorithmen ausgewählt wurde
    if 'All' in checked_labels:
        # remove 'All' from unique_labels
        checked_labels = list(set(unique_labels) - set(['All']))

    selected_domain = request.form.get('selected_domain')
    if selected_domain == 'all':
        selected_domain = None

    checked_algos = request.form.getlist('algorithm')
    if 'All' in checked_algos:
        checked_algos = None

    print('\n***CONFIGURATIONS***\n',
          'Window Size (Rows): ', window_size, '\n',
          'Filepath: ', filepath, '\n',
          'Abtastfrequenz (Hz): ', hertz, '\n',
          'Feature Domäne: ', selected_domain, '\n',
          'Checked labels: ', checked_labels, '\n',
          'Train Users: ', train_users, '\n',
          'Test Users: ', test_users, '\n',
          'Algorithmen: ', checked_algos, '\n\n')

    # Initialisiere ML_Tool-Instanz
    ml_tool = ML_Tool(window_size=window_size,
                      filepath=filepath,
                      hertz=hertz,
                      feature_domain=selected_domain,
                      labels=checked_labels,
                      train_users=train_users,
                      test_users=test_users,
                      algorithms=checked_algos,
                      feature_select=selection,
                      save_data=save_data)

    return ml_tool


# Active Learning Trainings mithilfe des Active Learning Tools (active-ml)
@app.route('/execute_al', methods=['POST'])
def execute_al():
    '''
        Führt Active Learning-Training für das Active Learning-Tool durch. Verarbeitet eingehende Streaming-Daten, 
        bewertet deren Nützlichkeit für das Modelltraining, fragt Benutzer bei Bedarf nach Label für Streaming-Daten,
        trainiert Modell solange, bis Abbruchbedingungen erreicht, speichert neues Modell.
        Aufruf, wenn ein POST-Request auf der Route '/execute_al' empfangen wird.
                Returns:
                        response (JSON): Ergebnisse des Trainings werden in einem JSON-Objekt zurückgegeben.
    '''
    # Initialisiere Status
    global status_al
    status_al.update({"status": "Ausführung", "finished": "", "error": "", "iteration": 0, "accuracy": 0, "uncertainty": 0})
    sample_counter = 0

    # Ausgewählten Modellnamen abrufen und Modell sowie Informationen laden
    selected_model = request.form['selected_model']
    model_filename = os.path.join(model_dir, selected_model)
    training_information, ml_model, model_name, classes, data_saved, X_train, y_train, X_test, y_test = load_model(
        model_filename)
    model, score, time_created, window_size, domain, labels = training_information
    window_size = int(window_size)

    # Wenn Benutzer nicht in Datenbank verfügbar, stoppe den Prozess
    user_id = int(request.form.get('user_id'))
    user_in_db = check_user(user_id)

    # Wenn keine Train-Test-Daten gespeichert wurden, stoppe den Prozess
    if data_saved == 'keine' or user_in_db == False or "SLKT" in score:
        # Aktuelle Zeit in Sekunden
        current_time = time.time()
        # Zeit in Struktur umwandeln
        time_struct = time.localtime(current_time)
        # Stunden, Minuten und Sekunden aus Struktur extrahieren
        hours = time_struct.tm_hour
        minutes = time_struct.tm_min
        seconds = time_struct.tm_sec
        # Aktuelle Zeit in lesbarem Format Stunden:Minuten:Sekunden
        current_time = "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)

        # Übermittle die Ergebnisse mittels Flash-Nachrichten, die im Template gerendert werden
        flash(Markup(f"<h2>Ergebnisse ({current_time})</h2>"))
        flash(Markup("<hr>"))

        # Rufe alle geflashten Nachrichten ab
        messages = get_flashed_messages()
        # Erstelle und sende JSON-Response zurück
        response = jsonify({'messages': messages})
        response.headers.add('Content-Type', 'application/json')

        if data_saved == 'keine':
            print("Ohne Train-Test-Daten kein AL für das ausgewählte Modell möglich.")
            status_al.update({"status": "Fertig", "finished": "", "error": "Achtung: Ohne Train-Test-Daten kein Active Learning für das ausgewählte Modell möglich."})
        elif "SLKT" in score:
            print("Achtung: Kein Active Learning für Modelle mit Feature Selektion möglich.")
            status_al.update({"status": "Fertig", "finished": "", "error": "Achtung: Kein Active Learning für Modelle mit Feature Selektion möglich."})
        else:
            print("Achtung: User ist nicht in der Datenbank verfügbar. Bitte prüfen Sie ihre ID.")
            status_al.update({"status": "Fertig", "finished": "", "error": "Achtung: User ist nicht in der Datenbank verfügbar. Bitte prüfen Sie ihre ID."})
        return response
    else:
        # Führe AL-Verfahren durch
        print('Starte AL.')
        status_al["status"] = "Starte AL - bitte Status beachten"

        time.sleep(2)
        
        # Messe Ausführungszeit
        start_time = time.time()
        
        # Initialisiere Paramter für AL-Verfahren und extrahiere Konfigurationen
        X_train = X_train
        y_train = y_train
        X_test = X_test
        y_test = y_test
        window_size = window_size
        domain = domain
        hertz = 20
        labels = classes

        max_iterations = int(request.form.get('max_iterations'))

        accuracy_threshold_str = request.form.get('accuracy_threshold')
        accuracy_threshold = float(accuracy_threshold_str.replace(',', '.'))

        uncertainty_threshold_str = request.form.get('uncertainty_threshold')
        uncertainty_threshold = float(uncertainty_threshold_str.replace(',', '.'))

        test_mode = request.form.get("test_mode")
        query_strategy = request.form.get('selected_strategy')

        # Wähle Query-Strategie
        if query_strategy == "Uncertainty Sampling":
            query_strategy = uncertainty_sampling
            print("Uncertainty Sampling")
        elif query_strategy == "Entropy Sampling":
            query_strategy = entropy_sampling
            print("Entropy Sampling")
        else:
            query_strategy = uncertainty_sampling


        # Intialisiere Aktiven Lerner
        learner = ActiveLearner(
            estimator = ml_model,
            query_strategy = query_strategy
        )

        # Stelle Verbindung zur Datenbank her
        db_service = Database_Service()

        # Berechne sleep_time für das Warten auf neue Daten
        sleep_time = float(window_size) / hertz

        # Berechne initiale Genauigkeit des Klassifikators (optional)
        status_al["status"] = "Berechne initiale Genauigkeit"
        unqueried_score = learner.score(X_test.values, y_test.values)*100
        print('Initial prediction accuracy: %.2f' % unqueried_score + ' %.')
        status_al["accuracy"] = round(unqueried_score, 2)

        time.sleep(3)

        # Aktuelle Zeit in Sekunden
        current_time = time.time()
        # Zeit in Struktur umwandeln
        time_struct = time.localtime(current_time)
        # Stunden, Minuten und Sekunden aus Struktur extrahieren
        hours = time_struct.tm_hour
        minutes = time_struct.tm_min
        seconds = time_struct.tm_sec
        # Aktuelle Zeit in lesbarem Format Stunden:Minuten:Sekunden
        current_time = "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)

        flash(Markup(f"<h2>Ergebnisse ({current_time})</h2>"))
        flash(Markup("<hr>"))
        flash(Markup("<h3>Initialgenauigkeit:</h3>"))
        flash(f'{unqueried_score:.2f} % Initialgenauigkeit.')
        #flash('Initial prediction accuracy: %.2f' % unqueried_score + ' %.')
        flash(Markup("<hr>"))
        flash(Markup("<h3>Trainingsverlauf:</h3>"))

        if test_mode is not None:
            # TEST MODUS - Pool für Simulationsmodus initialisieren
            X_pool = X_test.values
            print(X_pool)
        else:
            # STREAM MODUS - Initialisiere ML_Tool-Instanz
            ml_tool = ML_Tool(window_size=window_size,
                              hertz=hertz,
                              feature_domain=domain,
                              labels=labels)
            
        # Schleifen-Parameter zurücksetzen
        accuracy = 0
        iteration = 1

        # Trainiere Modell in einer Schleife, bis Abbruchbedingung erreicht ist
        while accuracy < accuracy_threshold and iteration < max_iterations+1 and status_al['finished'] == "":

            if test_mode is not None:
                # TEST MODUS - Zufallsstichprobe aus dem Simulationspool auswählen
                status_al["status"] = "Bitte führen Sie jetzt eine Aktivität aus (Simulationsmodus!)."
                print("Bitte führen Sie jetzt eine Aktivität aus (Simulationsmodus!).")
                stream_idx = np.random.randint(len(X_pool))
                stream_sample = X_pool[stream_idx].reshape(1, -1)
                # print("No. {iteration} - Random Sample: ", stream_sample)
                time.sleep(4)
            else:
                # STREAM MODUS - Warte auf neue Daten
                status_al["status"] = f'Bitte führen Sie jetzt eine Aktivität für {sleep_time} Sekunden aus.'
                print(f'Bitte führen Sie jetzt eine Aktivität für {sleep_time} Sekunden aus.')
                time.sleep(sleep_time)

                # Neueste Daten für User gemäß Window Size holen
                raw_data = db_service.get(user_id, window_size)
                print(f'No. {iteration} - Raw Sample: {raw_data}')

                # Extrahiere Features gemäß Feautre Domain
                stream_sample = ml_tool.auto_feature_extraction(raw_data)
                # print(f'No. {iteration} - Stream Sample: {stream_sample}')

            # Berechnung der Unsicherheit der Streamingdaten
            status_al.update({"status": "Berechne Nützlichkeit der Daten", "iteration": iteration})
            
            print("Berechne Nützlichkeit")
            if query_strategy == "Uncertainty Sampling":
                uncertainty = classifier_uncertainty(learner, stream_sample)
            elif query_strategy == "Entropy Sampling":
                uncertainty = classifier_entropy(learner, stream_sample)
            else:
                uncertainty = classifier_uncertainty(learner, stream_sample)

            print(f'Uncertainty: {uncertainty.item():.2f}')
            status_al["uncertainty"] = f'{uncertainty.item():.2f} für Iteration {iteration}.'

            time.sleep(5)

            # Wenn die Unsicherheit den Schwellenwert erreicht, Benutzer nach Label fragen
            if uncertainty >= uncertainty_threshold:
        
                status_al["status"] = "Bitte geben Sie den Aktivitätsnamen an!"
                print("Bitte geben Sie den Aktivitätsnamen an!")

                # Warte auf Benutzereingabe solange kein Label und kein Abbruch
                #label = input('Label: ')
                global label
                label = None
                while label is None and status_al['finished'] == "":
                    time.sleep(1)

                # Prüfe, ob Nutzer Abbruch erwirkt hat
                # Wenn ja, fahre fort, wenn nein, speichere Label und trainiere Modell neu
                if status_al['finished'] != "":
                    print("Abbruch, speichere Modell ohne erneut zu trainieren.")
                else:
                    # Feedback, welches Label zur Verfügung gestellt wurde
                    print(f'Label entered by user: {label}')
                    status_al["status"] = f'Verarbeite das Label "{label}". Trainiere Modell neu.'

                    # Füge Label und zugehörige Daten dem Trainingsdatensatz hinzu
                    X_train = np.concatenate([X_train, stream_sample])
                    y_train = np.concatenate([y_train, np.array([label])])
                    sample_counter += 1

                    # Trainiere den aktiven Lerner neu mit den hinzugefügten Trainingsdaten
                    learner.teach(X_train, y_train)
                    print('Labeled instance added to the training set.')

                    time.sleep(4)

                    # Berechne die neue Genauigkeit des Modells
                    accuracy = learner.score(X_test.values, y_test.values)
                    status_al["iteration"] = iteration
                    status_al["accuracy"] = round(accuracy*100, 2)
                    print(f'Iteration {iteration}: Accuracy = {accuracy:.2f}')

                    flash(f'{accuracy*100:.2f} % Genauigkeit – {iteration}. Interation – Label neu: {label}.')
                
            iteration += 1

        # Schließe Datenbankverbindung
        db_service.close()

        flash(Markup("<hr>"))

        # Bereite Daten für neuen Dateinamen vor
        name_short = model_name[:3]
        first_letters = ''.join([label[0] for label in labels])
        new_current_time = "{:02d}{:02d}{:02d}".format(hours, minutes, seconds)
        
        # Dateinamen erstellen und Ergebnisse ausgeben
        if accuracy < 0.01:
            filename = f"AL-{name_short}_{unqueried_score:.2f}_{new_current_time}_{window_size}_{domain}_{first_letters}.joblib"
            status_al["error"] = f"Aktives Lernen nach {iteration-1} Iterationen mit einer Genauigkeit von {unqueried_score:.2f} % beendet. {sample_counter} Stichprobe(n) zu Trainings-Daten hinzugefügt."
            flash(Markup("<h3>Finale Genauigkeit:</h3>"))
            flash(f'{unqueried_score:.2f} % Erkennungsgenauigkeit für {selected_model}.')
            flash(f'Modell gespeichert als: {filename}.')
            flash(Markup("<hr>"))
        else:
            filename = f"AL-{name_short}_{accuracy*100:.2f}_{new_current_time}_{window_size}_{domain}_{first_letters}.joblib"
            status_al["error"] = f"Aktives Lernen nach {iteration-1} Iterationen mit einer Genauigkeit von {accuracy*100:.2f} % beendet. {sample_counter} Stichprobe(n) zu Trainings-Daten hinzugefügt."
            flash(Markup("<h3>Finale Genauigkeit:</h3>"))
            flash(f'{accuracy*100:.2f} % Erkennungsgenauigkeit für {selected_model}.')
            flash(f'Modell gespeichert als: {filename}.')
            flash(Markup("<hr>"))
        
        # Speichere Model im angegebenen Verzeichnis unter dem erstellen Dateinamen
        file_path = os.path.join(model_dir, filename)
        joblib.dump((learner.estimator, X_train, y_train, X_test, y_test), file_path)

        print('Active learning stopped.')
        status_al["status"] = "Fertig"
        status_al['finished'] == ""

        print(
            f"Active Learning terminated after {iteration} iterations with accuracy of {accuracy:.2f}.")
        
        # Erstelle einen Klassifizierungsbericht für das neue AL-Modell
        learner.estimator.fit(X_train, y_train.ravel())
        y_predict = learner.estimator.predict(X_test)
        report = classification_report(y_test, y_predict, target_names=labels, zero_division=1)
        flash(Markup("<h3>Report zum neuen AL-Modell:</h3>"))
        flash(report)
        flash(Markup("<hr>"))
        
        # Messe Ausführungszeit
        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time:", execution_time, "seconds")

        flash(Markup("<h3>Ausführungszeit:</h3>"))
        flash(f"{execution_time:.2f} sec")
        flash(Markup("<hr>"))

        # Rufe alle geflashten Nachrichten ab
        messages=get_flashed_messages()
        # Erstelle und sende JSON-Response zurück
        response=jsonify({'messages': messages})
        response.headers.add('Content-Type', 'application/json')
        return response


# Flask-Route für Abbruch des AL-Verfahrens durch Benutzer.
@app.route('/update_status', methods=['POST'])
def update_status():
    '''
        Erhalte POST Request, um Status des AL-Verfahrens auf Abbrechen zu setzen. Initiiert Speichern und Beenden.
        Aufruf, wenn ein POST-Request auf der Route '/update_status' empfangen wird.
                Returns:
                        Ein JSON-Objekt, das den Erfolg der Operation angibt.
    '''
    global status_al
    data = request.get_json()
    status_al['finished'] = data['finished']
    return jsonify({'success': True})


# Flask-Route für das Abrufen des aktuellen Status im AL-Verfahren
@app.route('/get_status')
def get_status():
    '''
        Gibt aktuellen Status des AL-Verfahrens zurück.
        Aufruf, wenn ein GET-Request auf der Route '/get_status' empfangen wird. 
                Returns:
                        Ein JSON-Objekt, das den aktuellen Status enthält.
    '''
    global status_al
    return jsonify(status_al)


# Flask-Route für den Empfang des Labels des Probanden im AL-Verfahren
@app.route('/submit_label', methods=['POST'])
def submit_label():
    '''
        Extrahiert das Label aus HTML, speichert es global.
        Aufruf, wenn ein POST-Request auf der Route '/submit_label' empfangen wird. 
                Returns:
                        '': leerer String - es soll nichts passieren.
    '''
    global label
    label = request.form['label_text']
    return ''


# Prüfe, ob ein Benutzer in der Datenbank vorhanden ist
def check_user(user_id):
    print("Suche in Datenbank nach Benutzer")
    db_service = Database_Service()
    if db_service.check_user(user_id):
        print(f'User {user_id} ist in der Datenbank verfügbar!')
        db_service.close()
        return True
    else:
        print(f'User {user_id} ist !NICHT! in der Datenbank verfügbar!')
        db_service.close()
        return False


# Starten des Flask-Webservers in einem neuen Prozess
if __name__ == "__main__":
    with Manager() as manager:
        # Konfigurieren des Webserver-Hosts und -Ports
        app.run(host="0.0.0.0", port=5009)
