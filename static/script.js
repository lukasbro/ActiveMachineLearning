/* JavaScript für index.html und active-ml.html */
/* Erstellt von Lukas Bröning im Jahr 2023. */

// Toggle Switch Klassen verändern nach Klick
const switch1 = document.getElementById('mode1');
const switch2 = document.getElementById('mode2');

// Toggle Switch 1
switch1.addEventListener('click', () => {
  switch1.classList.add('active');
  switch2.classList.remove('active');
});

// Toggle Switch 2
switch2.addEventListener('click', () => {
  switch2.classList.add('active');
  switch1.classList.remove('active');
});

// Zum Ende der Seite scrollen
function scrollDown() {
  window.scrollTo({
    top: document.documentElement.scrollHeight,
    behavior: 'smooth'
  });
}