# Datenanalysen für Traveling Salesman Probleme

Diese Projekt enthält Ergebnisses eins Datenanalyse-Prokjekts zu Einfluss der Solver, TSP-Instanztypen und Solver-Konfigution zu Laufzeit eines gegebene Tsp Instanz.

## Ziel des Projekts
Mit 95% Wahrscheinlichkeit sagen, wann eine gegebene TSP-instanz fertig gelöst wird.

## Inhalt
-**Datenaufbereitung** Es wird hier Erte Statischen Maßen wie Mittelwert, Varianz , Median,.... Daten gleiche Intanzen berechnet, um Allgemeine Vwrhalten der Datensatz zu beobachten.
-**Modellierung**  Auswahl eine passende Modell zur Vorhersage und zum Lesen der Laufzeit. ==> Log-Linear-regression
-**Modell Bewertung** Hier wird überprüft, ob die 95% der Testdatennsatz echt unterhalb die Regressiongerade liegt und damitunsere Hypothese erfüllt.
-**Bestimmung der Abhängigkeit der Laufzeit von der Präprocessing Güte**  hierbei wird beobachtet, ob Präprocessing eien Einfluss auf die Laufzet von Intenzen hat.
-**Anpassung der Solver Konfiguration** Damit versuchen wir die Laufzeit zu beschleugnigen, wobei wir Solver Konfiguration anpassen, um Beispielweise passende Fensterbreite je
Intanzgröße zu finden und somit eine bessere Laufzeit zu schaffen ==> Optimierungsmöglichkeit

## Notebooks
Die Analysen sind meheren Datein organisiert.
- Zufällig und Regelmäßig.ipynb ==> enthaält Code zur Analyse von Zufälligen, Erzeugung und Analyse Regelmäßigen  TSP-Instanztypen
-  geklusterte Punkte-runtime, geklusterte_punkte_erzeugung, Beste_fensterbreit,
-  model-evaluation, plot_quantil_runtime stehen code zur Ergeugung , analyseder Geklustere Instanztypen, sowie Analyse der Beste Fensterbreite und teil Analyse der Komilitone Behe zur Zufälligen Instanztypen.


## Autor:innen
- Linelle Fontelle, Meneckdem Medawe 
- Mohamed Behe,  Reguigui
