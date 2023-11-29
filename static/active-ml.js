/* JavaScript für active-ml.html */
/* Erstellt von Lukas Bröning im Jahr 2023. */

/* Status des AL-Verfahren kontinuierlich aktualisieren */
function updateStatus() {
  var xhr = new XMLHttpRequest();
  xhr.open('GET', '/get_status');
  xhr.onload = function () {
    // Lese JSON-Response aus und aktualisiere den Status
    if (xhr.status === 200) {
      var response = JSON.parse(xhr.responseText);

      var statusText = response.status;
      var finished = response.finished;
      var iteration = response.iteration;
      var accuracy = response.accuracy;
      var uncertainty = response.uncertainty;
      var error_message = response.error;

      var statusElement = document.getElementById('status');
      var iterationElement = document.getElementById('iteration');
      var accuracyElement = document.getElementById('accuracy');
      var uncertaintyElement = document.getElementById('uncertainty');
      var error_messageElement = document.getElementById('error');

      statusElement.textContent = statusText;
      iterationElement.textContent = iteration;
      accuracyElement.textContent = accuracy;
      uncertaintyElement.textContent = uncertainty;
      error_messageElement.textContent = error_message;
    }
  };
  xhr.send();
}


/* Aktualisiere den Status des AL-Verfahrens jede Sekunde nachdem die Seite geladen wurde */
document.addEventListener('DOMContentLoaded', function () {
  var statusInterval = setInterval(updateStatus, 300);
});


/* Verwalte das Statuselement: animiere es und zeige bei spezifischem Status die Label-Buttons an */
var statusElement = document.getElementById('status');
var labelButtonsElements = document.getElementsByClassName('custom-button');
statusElement.addEventListener('DOMSubtreeModified', function () {
  var content = statusElement.textContent;
  var animatedClass = 'animated';
  var redClass = 'red';
  var greenClass = 'green';

  // Prüfe, ob der Inhalt nicht "Vorbereitung" oder "Fertig" ist, füge Klasse "animated" hinzu
  if (content !== 'Vorbereitung' && content !== 'Fertig') {
    if (!statusElement.classList.contains(animatedClass)) {
      statusElement.classList.add(animatedClass);
    }
  } else {
    // Andernfalls entferne die Klasse "animated", falls sie existiert.
    if (statusElement.classList.contains(animatedClass)) {
      statusElement.classList.remove(animatedClass);
    }
  }

  // Prüfe, ob der Inhalt "Bitte geben Sie den Aktivitätsnamen an!" lautet.
  if (content === 'Bitte geben Sie den Aktivitätsnamen an!') {
    // Zeige die Label-Buttons
    for (var i = 0; i < labelButtonsElements.length; i++) {
      labelButtonsElements[i].style.display = 'inline-block';
    }

    // Hinzufügen der Klasse "red" zum Statuselement
    if (!statusElement.classList.contains(redClass)) {
      statusElement.classList.add(redClass);
    }

    // Ans Ende der Seite scrollen
    scrollDown();

  } else {
    // Ausblenden der Label-Buttons
    for (var i = 0; i < labelButtonsElements.length; i++) {
      labelButtonsElements[i].style.display = 'none';
    }

    // Entferne Klasse "red" vom Statuselement
    if (statusElement.classList.contains(redClass)) {
      statusElement.classList.remove(redClass);
    }
  }

  // Prüfe, ob der Inhalt der Anforderung eines Labels entspricht.
  if (content.includes('jetzt')) {
    // Hinzufügen der Klasse "green" zum Statuselement
    if (!statusElement.classList.contains(greenClass)) {
      statusElement.classList.add(greenClass);
    }

    // Ans Ende der Seite scrollen
    scrollDown();
    
  } else {
    // Entferne Klasse "green" vom Statuselement
    if (statusElement.classList.contains(greenClass)) {
      statusElement.classList.remove(greenClass);
    }
  }
});


/* Führe das AL-Verfahren durch, steuere Bedienung von Elementen und gib Ergebnisse aus */
function execute_al_learning() {

  // Steuere Bedienung und Sichtbarkeit von Elementen
  $('.disable').prop('disabled', true);
  var hiddenButton = document.getElementById('save');
  hiddenButton.style.display = 'inline';
  document.getElementById('message-container').innerHTML = "";

  // Lese Werte der Eingabefelder aus
  var max_iterations = parseInt($('#max_iterations').val());
  var accuracy_threshold = parseFloat($('#accuracy_threshold').val());
  var uncertainty_threshold = parseFloat($('#uncertainty_threshold').val());
  console.log(max_iterations);
  console.log(accuracy_threshold);
  console.log(uncertainty_threshold);

  var errorMessage = "";

  // Prüfe die Zulässigkeit der Werte
  if (isNaN(accuracy_threshold) || accuracy_threshold < 0.01 || accuracy_threshold > 1.00 ||
    isNaN(max_iterations) || max_iterations < 1 || max_iterations > 100 ||
    isNaN(uncertainty_threshold) || uncertainty_threshold < 0.01 || uncertainty_threshold > 1.00) {
    errorMessage = 'Bitte geben Sie einen gültigen Zielgenauigkeits- und Nützlichkeitsschwellenwert zwischen 0.01 und 1.00 und eine gültige maximale Anzahl von Iterationen zwischen 1 und 100 an.';
    console.log(errorMessage);
  }

  // Wenn fehlerhafte Werte vorliegen, zeige den Fehler an und beende AL
  if (errorMessage.length > 0) {
    alert(errorMessage);
    console.log('ERROR');

    // Steuere Bedienung und Sichtbarkeit von Elementen
    $('.disable').prop('disabled', false);
    hiddenButton.style.display = 'none';
    return;
  }

  $.ajax({
    url: '/execute_al',
    type: 'POST',
    data: $('form').serialize(),
    success: function (response) {
      // Verarbeite Ergebnisse des AL-Verfahrens und stelle sie dar
      var messages = response.messages;
      var messageContainer = $('#message-container');
      // Leere Message Container
      messageContainer.empty();
      for (var i = 0; i < messages.length; i++) {
        // Wenn es sich um einen Report handelt, stelle ihn wie folgt dar
        if (messages[i].includes('precision')) {
          // Erstelle Tabelle für den Klassizierungsbericht
          var reportTable = $('<table id="report-table">');
          var reportLines = messages[i].split('\n');
          // Tabellen Header
          var headerRow = $('<tr>');
          headerRow.append($('<th>').text(''));
          headerRow.append($('<th>').text('precision'));
          headerRow.append($('<th>').text('recall'));
          headerRow.append($('<th>').text('f1-score'));
          headerRow.append($('<th>').text('support'));
          reportTable.append(headerRow);
          // Tabellen Daten
          for (var j = 2; j < reportLines.length - 5; j++) {
            var lineData = reportLines[j].trim().split(/\s+/);
            var dataRow = $('<tr>');
            dataRow.append($('<td>').text(lineData[0]));
            dataRow.append($('<td>').text(lineData[1]));
            dataRow.append($('<td>').text(lineData[2]));
            dataRow.append($('<td>').text(lineData[3]));
            dataRow.append($('<td>').text(lineData[4]));
            reportTable.append(dataRow);
          }
          // Füge Rest hinzu
          var additionalRows = reportLines.slice(reportLines.length - 5, reportLines.length);
          for (var j = 0; j < additionalRows.length; j++) {
            var lineData = additionalRows[j].trim().split(/\s+/);
            var dataRow = $('<tr>');
            dataRow.append($('<td>').text(lineData[0]));
            dataRow.append($('<td>').attr('colspan', '4').text(lineData.slice(1).join(' ')));
            reportTable.append(dataRow);
          }
          // Tabelle dem Message Container anhängen
          messageContainer.append(reportTable);
        }
        else {
          // Alle anderen Nachrichten werden wie folgt dargestellt.
          var message = $('<div>').html(messages[i]);
          messageContainer.append(message);
        }
      }
      // Steuere Bedienung und Sichtbarkeit von Elementen
      $('.disable').prop('disabled', false);
      hiddenButton.style.display = "none";
      document.getElementById('save-text').style.display = 'none';
      scrollDown();
    }
  });
}


/* ML-Modell löschen, bestätigen lassen */
function confirmDelete(model) {
  if (confirm('Sind Sie sicher, dass Sie das Modell ' + model + ' löschen wollen?')) {
    document.getElementById('delete_model').submit();
  }
}


/* AL-Verfahren abbrechen, speichern und beenden */
function setState(newState) {
  document.getElementById('save').style.display = 'none';
  document.getElementById('save-text').style.display = 'inline';

  // Sende POST Request, um Status auf Abbrechen zu setzen
  fetch('/update_status', {
    method: 'POST',
    body: JSON.stringify({ finished: newState }),
    headers: {
      'Content-Type': 'application/json'
    }
  }).then(response => {
    console.log('Status erfolgreich aktualisiert');
  }).catch(error => {
    console.error('Fehler bei der Aktualisierung des Status', error);
  });
}


/* Timer, der die Zeit für die aktuelle Aktivitätsausführung anzeigt */
function countdown(seconds, timerDiv, onFinish) {
  var act_interval = setInterval(function() {
    // Aktualisiere Timer, solange Zeit übrig
    if (seconds > 0) {
      timerDiv.textContent = 'Noch ' + seconds.toFixed(1) + ' Sekunden verbleibend';
      seconds -= 0.1;
    } else {
      // Stoppe Intervall sonst
      clearInterval(act_interval);
      timerDiv.parentNode.removeChild(timerDiv);
      onFinish();
    }
    // Aktualisiere alle 100 ms
  }, 100);
}

var countingDown = false;
// Beobachte Veränderungen am Statuselement
var observer = new MutationObserver(function(mutationsList) {
  var statusDiv = document.getElementById('status');
  if (statusDiv && !countingDown) {
    var text = statusDiv.textContent;
    // Prüfe, ob eine Zahl zum Herunterzählen im Status vorliegt
    var match = text.match(/\d+\.\d+/);
    if (match) {
      var seconds = parseFloat(match[0]);
      // Erstelle ein neues HTML-Element, in dem der Countdown-Timer angezeigt wird
      var timerDiv = document.createElement('div');
      timerDiv.id = 'timer';
      timerDiv.style.fontSize = '18px';
      statusDiv.parentNode.insertBefore(timerDiv, statusDiv.nextSibling);
      countingDown = true;
      // Rufe die countdown-Funktion auf, um den Timer zu starten
      countdown(seconds, timerDiv, function() {
        countingDown = false;
      });
    }
  }
});
// Starte Status-Observer, um auf Veränderungen zu horchen
observer.observe(document.body, { childList: true, subtree: true });
