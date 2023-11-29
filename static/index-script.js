/* JavaScript für index.html */
/* Erstellt von Lukas Bröning im Jahr 2023. */

/* Lies und prüfe die vom Benutzer eingegebenen Formulardaten, sende sie über AJAX an den Server. */
/* Initiiere ML-Modelltraining */
function execute_pre_feat() {
  // Werte aller Checkboxen auslesen
  var labelsChecked = $("input[name='label']:checked").length;
  var algosChecked = $("input[name='algorithm']:checked").length;
  var trainChecked = $('.train-checkbox:checked').length;
  var testCheckboxes = $('.test-checkbox');
  var testChecked = testCheckboxes.length > 0 ? testCheckboxes.filter(':checked').length : 1;

  // Werte für hertz und window_size auslesen und Zulässigkeit prüfen
  var hertz = parseInt($('#hertz').val());
  var window_size = parseInt($('#window_size').val());
  var errorMessage = "";
  if (isNaN(hertz) || isNaN(window_size) || hertz < 1 || hertz > 1000 || window_size < 1 || window_size > 1000) {
    errorMessage = 'Bitte geben Sie für Abtastfrequenz und Window Size gültige Zahlen zwischen 1 und 1000 ein.\n';
  }

  // Prüfe, ob für jede Checkbox-Kategorie mindestens eine Box markiert wurde
  var missingColumns = [];
  if (labelsChecked == 0) {
    missingColumns.push('Labels');
  }
  if (algosChecked == 0) {
    missingColumns.push('Algorithmen');
  }
  if (trainChecked == 0) {
    missingColumns.push('Train-Nutzer');
  }
  if (testChecked == 0) {
    if ($('.test-checkbox').length > 0) {
      missingColumns.push('Test-Nutzer');
    }
  }
  if (missingColumns.length > 0) {
    errorMessage += 'Bitte wählen Sie mindestens ein Kästchen in jeder Kategorie aus:\n- ' + missingColumns.join('\n- ');
  }

  // Rufe die ausgewählten Labels ab
  const label_checkboxes = document.querySelectorAll('input[name="label"]');
  const checked_labels = [];

  label_checkboxes.forEach(function (checkbox) {
    if (checkbox.checked && checkbox.value !== 'All') {
      checked_labels.push(checkbox.value);
    }
  });
  console.log(checked_labels);

  // Liefere ein Array von Objekten, die die ausgewählte Benutzer und deren verfügbaren Labels enthalten
  let checkedUsers = [];

  document.querySelectorAll('#user-table tbody tr').forEach((row) => {
    let trainCheckbox = row.querySelector('.train-checkbox');
    let testCheckbox = row.querySelector('.test-checkbox');

    if (trainCheckbox.checked || testCheckbox.checked) {

      // Ermittle Benutzer-ID und verfügbare Labels für diese Zeile
      let userID = row.querySelector('td:first-child').textContent.trim();
      let availableLabels = row.querySelector('td:last-child').textContent.trim();

      // Speichere Benutzer-ID und verfügbare Labels in einem Objekt
      let userLabels = { user: userID, labels: availableLabels.split(/\s*,\s*/) };

      // Objekt zum Array checkedUsers hinzufügen
      checkedUsers.push(userLabels);
    }
  });
  console.log(checkedUsers);

  // Stelle sicher, dass jedes ausgewählte Label für jeden ausgewählten Benutzer verfügbar ist
  checkedUsers.forEach(user => {
    checked_labels.forEach(label => {
      if (!user.labels.includes(label)) {
        // Erstelle Fehlermeldung, wenn ein ausgewähltes Label für einen ausgewählten Benutzer nicht verfügbar ist
        errorMessage += `Label '${label}' ist für Benutzer '${user.user} nicht verfügbar'\n`;
      }
    });
  });

  // Wenn Benutzeringaben fehlerhaft, zeige Fehler an und beende Training
  if (errorMessage.length > 0) {
    alert(errorMessage);
    return;
  }

  // Sende Formulardaten an den Server, da Validierung erfolgreich
  // Zeige Spinner an, während auf die Antwort gewartet wird. Steuere Bedienung und Sichtbarkeit von Elementen
  $('#spinner-container').show();
  $('button').prop('disabled', true);
  document.getElementById('message-container').innerHTML = "";

  $.ajax({
    url: '/ml_training',
    type: 'POST',
    data: $('form').serialize(),
    success: function (response) {
      // Verarbeite Ergebnisse des ML-Trainings und stelle sie dar
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
      $('#spinner-container').hide();
      $('button').prop('disabled', false);
      scrollDown();
    }
  });
}


/* Event Listener für die Änderungen von Train- und Test-Checkboxen beim Train-Test-Split */
$(document).ready(function () {
  $('.train-checkbox, .test-checkbox').on('change', function () {
    // Wenn eine Checkbox geändert wird, extrahiere die Anzahl der angekreuzten Train- und Test-Checkboxen
    var num_train = $('.train-checkbox:checked').length;
    var num_test = $('.test-checkbox:checked').length;
    var total = num_train + num_test;

    // Berechne Train-Test-Split
    var train_percent = 0;
    var test_percent = 0;

    // Prüfe Spezialfall, wenn keine User vorhanden, setze 80:20.
    if ($('.train-checkbox[data-user="None"]').length == 1) {
      train_percent = 80;
      test_percent = 20;
    } else {
      // Berechne Train-Test-Split in Prozent
      train_percent = Math.round(num_train / total * 100);
      test_percent = Math.round(num_test / total * 100);
    }

    // Aktualisiere Train-Test-Split-Element
    $('#train-test-split').text('Train: ' + train_percent + '%, Test: ' + test_percent + '%');
  });
});


/* Ermöglicht das De- / Aktivieren von mehreren Checkboxen durch eine Master-Checkbox */
function toggleCheckboxes(masterCheckbox, checkboxes) {
  // Wenn keine Boxen angegeben sind, beende
  if (!masterCheckbox || !checkboxes) {
    return;
  }

  // Füge Ereignislistener für das Ändern der Master-Checkbox hinzu
  masterCheckbox.addEventListener('change', function () {
    checkboxes.forEach(function (checkbox) {
      checkbox.checked = masterCheckbox.checked;
    });
  });
}


/* Verarbeite Nutzereingaben für die Label-Checkboxen */
// "All"-Label-Checkbox
const label_checkbox = document.querySelector('input[name="label"][value="All"]');
// Alle anderen Checkboxes
const label_checkboxes = document.querySelectorAll('input[name="label"]:not([value="All"])');
// Verwalte Checkboxeingaben
toggleCheckboxes(label_checkbox, label_checkboxes);

// Event Listener für jede Label-Checkbox
label_checkboxes.forEach(function (checkbox) {
  checkbox.addEventListener('change', function () {
    // Wenn sich eine der Label-Checkboxen ändert, prüfe, ob alle anderen Label-Checkboxen (außer der "Alle"-Checkbox) aktiviert sind. 
    // Wenn alle anderen Checkboxen aktiviert sind, wird die "Alle"-Checkbox ebenfalls aktiviert.
    const all_checked = Array.from(label_checkboxes).every(function (checkbox) {
      return checkbox.checked;
    });
    document.querySelector('input[name="label"][value="All"]').checked = all_checked;
  });
});


/* Verarbeite Nutzereingaben für die Algorithmen-Checkboxen */
// "All"-Algorithmen-Checkbox
const algo_checkbox = document.querySelector('input[name="algorithm"][value="All"]');
// Alle anderen Checkboxes
const algorithm_checkboxes = document.querySelectorAll('input[name="algorithm"]:not([value="All"])');
// Verwalte Checkboxeingaben
toggleCheckboxes(algo_checkbox, algorithm_checkboxes);

// Event Listener für jede Algorithmen-Checkbox
algorithm_checkboxes.forEach(function (checkbox) {
  checkbox.addEventListener('change', function () {
    // Wenn sich eine der Algorithmen-Checkboxen ändert, prüfe, ob alle anderen Algorithmen-Checkboxen (außer der "Alle"-Checkbox) aktiviert sind. 
    // Wenn alle anderen Checkboxen aktiviert sind, wird die "Alle"-Checkbox ebenfalls aktiviert.
    const all_checked = Array.from(algorithm_checkboxes).every(function (checkbox) {
      return checkbox.checked;
    });
    document.querySelector('input[name="algorithm"][value="All"]').checked = all_checked;
  });
});


/* Verarbeite Nutzereingaben für die Train-Test-Checkboxen */
// Alle Checkboxen für Train-Test-Daten abrufen
const train_checkboxes = document.querySelectorAll('.train-checkbox');
const test_checkboxes = document.querySelectorAll('.test-checkbox');

// Event Listener für jede Train-Checkbox
train_checkboxes.forEach(function (train_checkbox) {
  train_checkbox.addEventListener('change', function () {
    // Wenn der Benutzer eine Train-Checkbox aktiviert, deaktiviere Test-Checkbox in der gleichen Zeile
    if (train_checkbox.checked) {
      const row = train_checkbox.closest('tr');
      const test_checkbox = row.querySelector('.test-checkbox');
      if (test_checkbox.checked) {
        test_checkbox.checked = false;
      }
    }
  });
});

// Event Listener für jede Test-Checkbox
test_checkboxes.forEach(function (test_checkbox) {
  test_checkbox.addEventListener('change', function () {
    // Wenn der Benutzer eine Test-Checkbox aktiviert, deaktiviere die Train-Checkbox in der gleichen Zeile
    if (test_checkbox.checked) {
      const row = test_checkbox.closest('tr');
      const train_checkbox = row.querySelector('.train-checkbox');
      if (train_checkbox.checked) {
        train_checkbox.checked = false;
      }
    }
  });
});

// Alphabetische Sortierung für Text-Elemente in den "Verfügbare Labels"-Zellen
const labelzellen = document.querySelectorAll('td.labels-verfuegbar');
labelzellen.forEach(td => {
  const zellentext = td.textContent;
  const textsplit = zellentext.split(',').map(text => text.trim());
  const zellesortiert = textsplit.sort().join(', ');
  td.textContent = zellesortiert;
});


/* Verfügbare Labels so anpassen, dass, wenn alle verfügbar sind, "alle" angezeigt wird */
/* -- Atuell nicht möglich, da Prüfmechanismus auf diese Elemente zugreifen muss --
var unique_labels = document.getElementById("unique_labels").dataset.uniqueLabels.split(",");

// "All"-Element und unnötige Zeichen entfernen
unique_labels.shift();
for (var i = 0; i < unique_labels.length; i++) {
    unique_labels[i] = unique_labels[i].trim().replace(/'/g, "");
    if (i === unique_labels.length - 1) {
        unique_labels[i] = unique_labels[i].replace("]", "");
    }
}

// Wenn ein td-Element alle Label enthält, ersetze den Inhalt entsprechend
function alleLabelsErsetzen() {
    const labelzellen = document.getElementsByClassName("labels-verfuegbar");

    for (var i = 0; i < labelzellen.length; i++) {
        var labeltext = labelzellen[i].innerText.trim();
        var labels = labeltext.split(",").map(function (label) {
            return label.trim();
        });
        if (unique_labels.every(label => labels.includes(label))) {
            labelzellen[i].innerText = "Alle";
        }
    }
}
window.addEventListener("load", alleLabelsErsetzen);
*/