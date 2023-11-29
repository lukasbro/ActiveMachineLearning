'''
ml_trainingstool.py
    Machine Learning Modelltraining: 
    Lade CSV-Datensatz, filtere Datensatz, führe Train-Test-Split durch, Feature Extraktion, Feature Selektion.
    Modelltraining und -test, Evaluation und Speichern des besten Algorithmus, Ausgabe der Ergebnisse.

    Created by Lukas Broening in 2023.
    Tested for:
        macOS Monterey Version 12.6
        Python Version 3.11.2
        Terminal Version 2.12.7 (445)
        pandas Version 1.5.3
        scikit-learn Version 1.2.1
        tsfel Version 0.1.5

        Linux-4.19.0-18-amd64-x86_64-with-debian-10.11
        Python Version 3.7.3
        Terminal Version 2.12.7 (445)
        pandas Version 1.3.1
        scikit-learn Version 0.24.2
        tsfel Version 0.1.4
        joblib Version 1.0.1
'''

# Bibliotheken-Packages-Import
import os
import tsfel
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis


class ML_Tool:

    # Konfiguration - Initialisierung
    def __init__(self, window_size=260,
                 filepath='/your_path/data/your_file.csv',
                 hertz=20,
                 feature_domain='statistical',
                 labels=['Trinken', 'Schreiben', 'Essen'],
                 train_users=['2', '6', '12'],
                 test_users=['11'],
                 algorithms=None,
                 feature_select=None,
                 save_data=None,
                 columns=['accelUserX', 'accelUserY', 'accelUserZ', 'attitudePitch', 'attitudeRoll', 'attitudeYaw',
                          'gravityX', 'gravityY', 'gravityZ', 'gyroX', 'gyroY', 'gyroZ', 'label', 'user_id'],
                 header=['accX', 'accY', 'accZ', 'attPi', 'attRo', 'attYa',
                         'gravX', 'gravY', 'gravZ', 'gyroX', 'gyroY', 'gyroZ']):

        # Window Size für die Segmentierung und Featureextraktion
        self.window_size = window_size

        # Dateiverzeichnis des Datensatzes
        self.filepath = filepath

        # Sampling Frequenz des Datensatzes
        self.hertz = hertz

        # Bereich für die automatische Extraktion von Zeitreihenmerkmalen: zeitlich, spektral, statistisch, None (Alle)
        self.feature_domain = feature_domain

        # Label-Liste nach der im Datensatz gefiltert werden soll
        self.labels = labels

        # User-Listen nach denen im Datensatz gefiltert werden soll
        self.train_users = train_users
        self.test_users = test_users

        # Liste von Spalten, nach denen im Datensatz gefiltert werden soll
        self.columns = columns

        # Header Namen für die Merkmalsextraktion, passend zu den Spalten (Sensordatentypen)
        self.header = header

        # Algorithmen-Liste
        self.algorithms_dict = {
            "GaussianNB": GaussianNB(),
            "LogisticRegression": LogisticRegression(n_jobs=1, C=1e5, solver='liblinear', penalty='l1', multi_class='ovr', random_state=0, max_iter=1000),
            "SVC": SVC(gamma='scale', kernel='rbf'),  # Takes a long time
            "DecisionTree": DecisionTreeClassifier(random_state=0),
            "KNN": KNeighborsClassifier(algorithm='brute', n_neighbors=10, p=1, weights='distance'),
            # Takes a long time
            "MLP": MLPClassifier(random_state=1, max_iter=500),
            "RandomForest": RandomForestClassifier(max_depth=2, random_state=0),
            "AdaBoostClassifier": AdaBoostClassifier(),
            "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis(reg_param=0.1),
            "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
            "SGDClassifier": SGDClassifier(), # NO AL
            "GradientBoost": GradientBoostingClassifier()
        }

        # Feature Selektion JA/NEIN
        self.feature_select = feature_select

        # Speicherung Train-Test-Daten JA/NEIN
        self.save_data = save_data

        # Prüfe ob Algo-Dictionary übergeben wurde und nutze dieses, nimm alle Algorithmen sonst
        if algorithms is not None:
            self.algorithms = {
                k: v for k, v in self.algorithms_dict.items() if k in algorithms}
        else:
            self.algorithms = self.algorithms_dict

    
    # Lade Datensatz aus CSV-Datei, filtere ihn gemäß der Konfigurationen, mache Train-Test- und Daten-Label-Split.
    def load_filter_data(self):
        '''
        Lädt den Datensatz, filtert ihn entsprechend den Konfigurationen und trennt die Label von den Daten.
                Returns:
                        X_train (DataFrame): Pandas-Dataframe, der Daten gemäß der Liste der Spalten und Label für Training-Nutzer enthält.
                        y_train (DataFrame): Pandas-Datenframe, der die mit X_train assoziierten Labels enthält.
                        X_test (DataFrame): Pandas-Datenframe, der Daten gemäß der Liste der Spalten und Label für Testbenutzer enthält.
                        y_test (DataFrame): Pandas-Dataframe, der die mit X_test assoziierten Labels enthält.
        '''

        # Lade die Sensor-Rohdaten in einen Pandas-Datenrahmen
        rawData = pd.read_csv(self.filepath)

        # Filtere den Dataframe basierend auf der Label- und Spaltenliste
        filtered_df = rawData.loc[rawData['label'].isin(self.labels) & rawData['user_id'].isin(self.train_users + self.test_users), self.columns]

        # Filtere den Dataframe basierend auf den Train/Test-Benutzern
        train_data = filtered_df.loc[filtered_df['user_id'].isin(self.train_users)]
        test_data = filtered_df.loc[filtered_df['user_id'].isin(self.test_users)]

        # Teile den Datensatz in Features (X) und Labels (y) auf
        X_train = train_data.drop(['label', 'user_id'], axis=1)
        y_train = train_data['label']

        X_test = test_data.drop(['label', 'user_id'], axis=1)
        y_test = test_data['label']

        return X_train, y_train, X_test, y_test


    # Teile Label-Datensatz in Segmente gemäß Window Size auf, extrahiere Features für den Datensatz.
    def auto_feature_extraction(self, df):
        '''
        Empfängt eine Zeitreihe, unterteilt sie in Segmente (Window Size z.B. 250: 5 * 50 ms = 5 s) und extrahiert automatisch Features
                Parameter:
                        df (DataFrame): Pandas-Datenframe mit Daten für Training oder Test.
                Returns:
                        df_feat (DataFrame): Pandas-Dataframe mit allen neu extrahierten Features für den Datensatz.
        '''

        # Statistical/spectral/temporal Merkmale werden extrahiert
        cfg_file = tsfel.get_features_by_domain(self.feature_domain)

        # Extrahiere Features entsprechend den Konfigurationen
        df_feat = tsfel.time_series_features_extractor(
            cfg_file, df, fs=self.hertz, window_size=self.window_size, header_names=self.header)
        
        print('\n \n')

        return df_feat


    # Teile Label-Datensatz in Segmente gemäß Window Size auf, berechne Dataframe mit häufigstem Label pro Segment.
    def extract_labels(self, df):
        '''
        Iteriere über einen Datensatz gemäß Window Size und liefere einen Datenrahmen mit dem häufigsten Label für jedes Segment.
                Parameter:
                        df (DataFrame): Pandas-Datenframe, der nur Labels enthält.
                Returns:
                        y_labels (DataFrame): Pandas-Datenframe mit häufigstem Label für jedes Segment.
        '''

        # Initialisiere eine leere Liste, um die häufigsten Label zu speichern
        most_common_labels = []

        # Lege die Anzahl der Zeilen fest, über die iteriert werden soll
        steps = self.window_size

        # Durchlaufe den Dataframe
        for i in range(0, len(df), steps):
            # Hole die aktuelle Teilmenge der zu berücksichtigenden Zeilen
            subset = df.iloc[i:i+steps]

            # Berücksichtige die Teilmenge nur, wenn sie mindestens #steps Zeilen enthält
            if len(subset) >= steps:
                # Hole die Anzahl der Werte für die Spalte "label" in der aktuellen Teilmenge
                label_counts = subset.value_counts()

                # Ermittle das am häufigsten vorkommende Etikett in der aktuellen Teilmenge
                most_common_label = label_counts.index[0]

                # Füge das am häufigsten verwendete Label der Liste hinzu
                most_common_labels.append(most_common_label)

        # Erzeuge einen neuen Datenframe mit den gebräuchlichsten Labels
        y_labels = pd.DataFrame({'label': most_common_labels})

        # print(y_labels)
        return y_labels


    # Feature-Selektion mithilfe Korrelationsberechnung und Entfernen hoch korrelierter Features.
    def feature_selection(self, x_train_feat, x_test_feat):
        '''
        Selektion von Features durch Berechnung der paarweisen Korrelation von Features mit der Pearson-Methode und Entfernen von hoch korrelierten Features.
                Parameter:
                        x_train_feat (DataFrame): Dataframe mit allen extrahierten Features für das Training.
                        x_test_feat (DataFrame): Datenframe mit allen extrahierten Features für den Test.
                Returns:
                        x_train_feat (DataFrame): Dataframe mit allen für das Training selektierten Features.
                        x_test_feat (DataFrame): Dataframe mit allen für den Test selektierten Features.
        '''

        # Alle Features zählen / darstellen
        get_OriginalFeatures = set(x_train_feat.columns)
        num_OriginalFeatures = x_train_feat.shape[1]
        '''
        print('\n', ' +++ Features Extracted +++ ',
              '\n', get_OriginalFeatures, '\n')
        '''
        print(' +++ Anzahl Features Extracted: ',
              '\n', num_OriginalFeatures, '\n')

        # Wenn Feature Selektion JA, führe sie durch
        if self.feature_select is not None:
            # Entferne hoch korrelierte Features
            corr_features = tsfel.correlated_features(x_train_feat)
            x_train_feat.drop(corr_features, axis=1, inplace=True)
            x_test_feat.drop(corr_features, axis=1, inplace=True)

            # Selektierte Features zählen / darstellen
            get_CorrelatedFeatures = set(x_train_feat.columns)
            num_Features = x_train_feat.shape[1]
            '''
            print(' +++ Features ohne Corr. Features +++ ',
                '\n', get_CorrelatedFeatures, '\n')
            '''
            print(' +++ Anzahl Features ohne Corr. Features: ',
                '\n', num_Features, '\n')

            # Entfernte Features zählen / darstellen
            get_DeletedFeatures = get_OriginalFeatures.difference(
                get_CorrelatedFeatures)
            num_DeletedFeatures = len(get_DeletedFeatures)
            '''
            print(' +++ Geloeschte Corr. Features +++ ',
                '\n', get_DeletedFeatures, '\n')
            '''
            print(' +++ Anzahl geloeschte Corr. Features: ',
                '\n', num_DeletedFeatures, '\n')

            # Normalisierung der Features (opt)
            '''
            scaler = preprocessing.StandardScaler()
            nX_train = scaler.fit_transform(x_train_feat)
            nX_test = scaler.transform(x_test_feat)
            '''

            return x_train_feat, x_test_feat
        
        # Wenn keine Feature Selektion ausgewählt, gib die Dataframes unbearbeitet zurück
        else:
            print(' +++ NO FEATURE SELECTION +++', '\n')
            return x_train_feat, x_test_feat


    # Trainiere und teste Klassifikatoren mit Trainings- und Testdaten.
    # Speichere Klassifikator mit der höchsten Genauigkeit in Datei. Gib Rangliste, Klassifikationsbericht und Genauigkeit zurück.
    def classify(self, X_train, X_test, y_train, y_test, current_time):
        '''
        Trainiere Klassifikatoren gemäß dem Algorithmus-Dictionary, sage Testdaten voraus, zeige die Ergebnisse, speichere das beste Modell.
                Parameter:
                        X_train (DataFrame): Pandas-Datenframe mit allen extrahierten Features für das Training.
                        X_test (DataFrame): Pandas-Datenframe mit allen extrahierten Features für den Test.
                        y_train (DataFrame): Pandas-Datenframe mit den häufigsten Label pro Segment für das Training.
                        y_test (DataFrame): Pandas-Datenframe mit den häufigsten Label pro Segment für den Test.
                Returns:
                        classifiers_message (String): Zeichenkette, die sortierte Klassifikatoren (Ranking) und deren Genauigkeiten auflistet.
                        report_message (String): Zeichenkette, die Klassifizierungsbericht des besten Algorithmus (Rang 1) enthält.
        '''

        # Erstelle eine leere Liste, um die Genauigkeit und den Namen jedes Klassifikators zu speichern
        results = []

        # Trainiere Klassifikator
        for name, algorithm in self.algorithms.items():
            try:
                print("Final X Train for ML-Training: ", X_train)
                algorithm.fit(X_train, y_train.ravel())

                # Testdaten vorhersagen
                y_predict = algorithm.predict(X_test)

                # Berechne Genauigkeit und Klassifizierungsbericht
                accuracy = accuracy_score(y_test, y_predict)*100
                report = classification_report(
                    y_test, y_predict, target_names=self.labels, zero_division=1)
                print(f"{name}: {accuracy:.2f} % Accuracy")
                print(report)

                # Extrahiere Informationen zum Training, darunter Klassen und Feature Domäne
                name_short = name[:3]
                first_letters = ''.join([label[0] for label in self.labels])

                if self.feature_domain == None:
                    domain = 'all'
                else:
                    domain = self.feature_domain

                # Erstelle den Dateinamen
                if self.feature_select is not None:
                    model_filename = f"{name_short}_SLKT-{accuracy:.2f}_{current_time}_{self.window_size}_{domain}_{first_letters}.joblib"
                else:
                    model_filename = f"{name_short}_{accuracy:.2f}_{current_time}_{self.window_size}_{domain}_{first_letters}.joblib"

                # Pfad zu dem Verzeichnis, in dem das Modell gespeichert werden soll
                save_dir = '/your_path/trained_models/'

                # Prüfe, ob das Verzeichnis existiert, wenn nicht, erstelle es
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # Verbinde das Verzeichnis und den Dateinamen mit os.path.join
                file_path = os.path.join(save_dir, model_filename)

                # Speichere das trainierte Modell (und die Daten) in der Datei mit dem angegebenen Dateinamen
                if self.save_data == None:
                    joblib.dump(algorithm, file_path)
                    print("Model gespeichert.")
                else:
                    joblib.dump((algorithm, X_train, y_train, X_test, y_test), file_path)
                    print("Model und Daten gespeichert.")

                # Füge die Genauigkeit und den Namen des Klassifikators der Ergebnisliste an
                results.append((accuracy, name, report))

            except Exception as e:
                print(f"{name} - Ausführung mit Fehler fehlgeschlagen: {e}")
                classifiers_message, report_message, filename = f"{name} - Ausführung mit Fehler fehlgeschlagen: {e}"
                return classifiers_message, report_message

        # Sortiere die Ergebnisliste in absteigender Reihenfolge basierend auf dem Genauigkeitswert
        results = sorted(results, reverse=True)

        # Printe die sortierte Liste der Klassifikatoren und ihre Genauigkeitswerte
        print("\nClassifiers ordered by accuracy:")
        for rank, result in enumerate(results):
            print(f"{rank+1}. {result[0]:.2f} % - {result[1]}")

        # Extrahiere die Namen, Genauigkeiten und Berichte aus der Ergebnisliste
        accuracies, classifier_names, reports = zip(*results)

        # Sortiere die Klassifikatoren nach Genauigkeit in absteigender Reihenfolge und füge den Rang hinzu
        classifiers_sorted = [(rank+1, accuracy, name) for rank, (accuracy, name) in enumerate(sorted(zip(accuracies, classifier_names), reverse=True))]

        # Ermittele den Namen des besten Klassifikators
        best_algorithm_name = classifiers_sorted[0][2]
        best_algorithm_name = best_algorithm_name[:3]

        # Gehe durch alle gespeicherten Dateien im Verzeichnis
        for filename in os.listdir(save_dir):
            # Prüfe, ob der Dateiname mit current_time übereinstimmt und nicht der beste Algorithmus ist
            if current_time in filename and best_algorithm_name not in filename:
                # Entferne die Datei
                os.remove(os.path.join(save_dir, filename))

        # Name und Bericht über den besten Algorithmus auf Platz 1
        best_report = reports[0]
        for filename in os.listdir(save_dir):
            if current_time in filename:
                best_algo_filename = filename
                

        # Erstelle String mit sortierter Klassifikator-Liste und Klassifizierungsbericht des besten Algorithmus
        classifiers_message = "<br>".join([f"{result[0]}. {result[1]:.2f} % - {result[2]}" for result in classifiers_sorted])
        report_message = f"<br>{best_report}"

        return classifiers_message, report_message, best_algo_filename