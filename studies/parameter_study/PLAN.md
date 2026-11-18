# Parameterstudie DLR-CMA-ES: Plan und Übergabe

Stand: 16.11.2026. Dieses Dokument enthält alles, um die Studie ohne den
bisherigen Gesprächsverlauf fortzusetzen. Lies es einmal ganz, bevor du etwas
startest.

## 1. Ziel

Der Nutzer (Masterarbeit) will den Algorithmus DLR-CMA-ES **besser verstehen**.
Konkret soll die Studie zeigen, ob die verbleibenden Schwächen des Algorithmus

- (a) am **Niedrigrang-Ansatz** liegen,
- (b) an den **Parameterheuristiken** (Rangwahl, Update-Intervall, Lernraten),
- (c) oder schon am **CMA-ES selbst**.

Es geht also nicht in erster Linie darum, bessere Standardwerte zu finden,
sondern Ursachen zuzuordnen. Jede Auswertung muss am Ende eine Aussage der
Form „Schwäche X auf Funktion Y liegt an (a), (b) oder (c)" ermöglichen.

## 2. Feste Regeln

1. **Nur in `studies/parameter_study/` arbeiten.** Der Ordner hieß bis
   18.11.2026 `tmp/` (gitignoriert); er wurde auf Wunsch des Nutzers in
   `studies/` umbenannt und liegt versioniert auf dem Branch
   `parameter-study` (Fork github.com/entuuu/seamaze). Commits und Pushes
   nur, wenn der Nutzer darum bittet; nie auf `master`.
2. **Die Bibliothek `seamaze/` nicht verändern.** Auch `examples/dlrcmaes.py`
   nicht anfassen (dort liegt eine eigene, nicht committete Änderung des
   Nutzers).
3. **Keine rotierten Zusatzfunktionen in der Hauptstudie.** Die Klasse
   `Rotated` in `study.py` und der Lauf `rotpilot` waren ein eigenmächtiger
   Zusatz; der Nutzer wollte ihn nicht. Er bleibt nur als Randnotiz
   (Abschnitt 6.4). Nur auf ausdrücklichen Wunsch wieder aufnehmen.
4. **Interne Konstanten des Solvers** (Abschnitt 3.3) nur nach Rückfrage
   untersuchen, und dann in einer Kopie des Solvers im Studienordner.
5. **Nie zwei Studienläufe gleichzeitig starten.** Der Rechner hat 8 Kerne,
   ein Lauf nutzt 6 Prozesse.
6. **Vor Läufen, die länger als etwa eine Stunde dauern, dem Nutzer Umfang und
   geschätzte Dauer nennen.** Lange Läufe im Hintergrund starten.
7. **Mit dem Nutzer auf Deutsch schreiben.** Ergebnisse knapp, mit Zahlen.
8. **Plots:** kein Panel und keine Linie, die nichts zeigt (siehe Abschnitt 9).
9. **Bei der Parameterstudie bleiben.** Der Nutzer hat am 17.11.2026 bemängelt,
   dass die Diagnosen (Abschnitt 7) zu weit vom Plan abgedriftet sind. Die
   Hauptlinie sind die Stufen 1–3 der Parameterstudie. Keine neuen
   Diagnosen ohne Rückfrage; Diagnosebefunde nur als kurze Erklärung nutzen.
10. **Vergleiche mit CMA-ES nur gegen die korrigierte Kopie**
    (`variants/cmaes_path_variant.py`, Label `CMA-ES korr.`), nie gegen
    `seamaze.optimizers.CMAES` (Fehler im Schrittweiten-Pfad, Abschnitt 7).

## 3. Der Algorithmus in Kürze

Quelle: `seamaze/optimizers/_dlrcmaes.py`, Klasse `DLRCMAES`.

Die Kovarianzmatrix wird zerlegt in C = U·S·Uᵀ + diag(ψ). U·S·Uᵀ ist der
Niedrigrang-Teil mit r Richtungen, ψ die Restvarianz pro Koordinate.

### 3.1 Öffentliche Parameter

| Parameter | Standard | Wirkung |
|---|---|---|
| `low_rank_energy_tolerance` | 1e-2 | Rangwahl: kleinster Rang, der Anteil (1 − Toleranz) der Energie des nicht durch ψ erklärten Spektrums abdeckt. Kleinere Toleranz → höherer Rang. |
| `low_rank_max_dimension` | n | Obergrenze für den Rang. |
| `low_rank_init_dimension` | 1 | Startrang. C ist am Anfang immer die Einheitsmatrix. |
| `low_rank_is_adaptive` | True | Bei False bleibt der Rang auf `low_rank_init_dimension`. |
| `update_interval` | automatisch | U, S, ψ werden nur jede k-te Generation aktualisiert. Automatik: ceil(√(r / (n·c_cov))), neu berechnet bei Rangänderung. Bei n = 100: Rang ≤ 3 → 1, ≤ 15 → 2, ≤ 35 → 3, ≤ 62 → 4, darüber 5. |
| `number_of_individuals` | 4 + int(3·ln n) | Populationsgröße, bei n = 100 also 17. |
| `initial_sigma` | 1.0 | Startschrittweite. |

Der Rang ändert sich pro Kovarianz-Update nur um ±1.

### 3.2 Abbruchkriterien (in der Studie festgehalten)

`tolerance`, `fitness_window_size`, `sigma_threshold`, `maximum_iterations`,
`maximum_wall_time`, `fitness_threshold`. Siehe Abschnitt 5.

### 3.3 Fest verdrahtete Konstanten (nicht öffentlich)

- `beta = 0.0` in `_get_learning_rates`: Lernraten × (n/r)^β. Mit 0 ohne
  Wirkung.
- `threshold = 0.02` in `_adaptive_bug_step`: Rauschfilter der Rangwahl.
- Rangschritt ±1, Krylov-Dimension 20.

### 3.4 Referenz: normales CMA-ES

`seamaze/optimizers/_cmaes.py`, Klasse `CMAES`. Gleiche Schnittstelle ohne die
`low_rank_*`-Parameter. Eigenes Standardintervall ceil(√(1/c_cov)), bei
n = 100 ist das 6. Zusätzliches Abbruchkriterium `NO_EFFECT_AXIS`.

## 4. Dateien im Studienordner

Ordnerstruktur seit 17.11.2026 (aufgeräumt; ältere Abschnitte nennen
teilweise noch die alten Pfade wie `results/stage1.csv`, gemeint ist dann
der neue Ort laut dieser Tabelle):

```
studies/
  schwefel_versuche/        eigene frühere Schwefel-Skripte des Nutzers (nicht Teil der Studie)
  parameter_study/
    PLAN.md                 dieses Dokument
    study.py                Einstellungen, Funktionen, alle Konfigurationen, Einzellauf run_single
    run_study.py            parallele Ausführung einer Stufe, fortsetzbar
    analysis/               Auswertungen (analyze_stage1.py, analyze_stage2.py, analyze_ladder.py)
    diagnostics/            Diagnose-Skripte diag_*.py (abgeschlossen, siehe Abschnitt 7)
    variants/               Diagnose-Kopien der Solver + Erzeugerskripte
    results/
      pilot/                pilot.csv, Auswertung, plots/
      stage1/               stage1.csv (900 Läufe), stage1_summary.csv, stage1_table.txt, plots/
      stage2/               stage2.csv (Raster Toleranz × Intervall), Auswertung, plots/
      ladder/               ladder.csv, ladder_ref.csv (CMA-ES korr.), psifold.csv, ladder_table.txt, plots/
                            plots/alt_fehlerhafte_referenz/: veraltete Plots mit fehlerhaftem CMA-ES, nicht verwenden
      diagnostics/          Ausgaben und Plots der Diagnose-Skripte
      archiv_rotpilot/      rotpilot.csv (vom Nutzer nicht gewollt, nur Randnotiz)
      logs/                 Konsolenausgaben, eine Datei pro Stufe
```

`study.py` hängt `variants/` an `sys.path`; Skripte in `analysis/` und
`diagnostics/` hängen den Studienordner und `variants/` selbst an. Alle
Skripte daher aus `studies/parameter_study/` heraus starten.

`variants/`: `cmaes_path_variant.py` (CMA-ES mit korrigiertem
Schrittweiten-Pfad, Referenz für alle Vergleiche; erzeugt von
`make_cmaes_variant.py`), `dlr_variant_ident.py` (unveränderte Kopie,
Identitätsprüfung), `dlr_variant_psifold.py` (ψ im Update); beide erzeugt
von `make_dlr_variants.py`. Nach einer Änderung an der Bibliothek neu
erzeugen.

### Befehle

Immer aus `studies/parameter_study/` heraus, mit dem Python der Umgebung:

```
cd studies/parameter_study
../../.venv/Scripts/python.exe run_study.py <stufe> <prozesse> <seeds> > results/logs/<stufe>.log 2>&1
../../.venv/Scripts/python.exe analysis/analyze_stage1.py stage1 > results/stage1/stage1_table.txt
../../.venv/Scripts/python.exe analysis/analyze_stage2.py
../../.venv/Scripts/python.exe analysis/analyze_ladder.py
../../.venv/Scripts/python.exe run_study.py repro
```

- `<stufe>`: `pilot`, `stage1`, `stage2`, `ladder`, `ladder_ref`, `psifold`,
  `rotpilot`. Ergebnis landet in `results/<ordner>/<stufe>.csv`
  (`ladder_ref` und `psifold` im Ordner `ladder`).
- `<prozesse>`: 6 verwenden.
- `<seeds>`: Anzahl; es laufen die Seeds 0 bis Anzahl − 1.
- `repro` prüft, dass zwei identische Läufe identische Ergebnisse liefern.
- `run_study.py` bittet Windows während des Laufs, nicht in den Ruhezustand
  zu gehen (`SetThreadExecutionState`, nur solange der Prozess läuft, keine
  Energieeinstellungen geändert). Zuklappen des Laptops kann trotzdem den
  Ruhezustand auslösen.

**Fortsetzen:** Denselben Befehl noch einmal starten. Fertige Läufe werden
über den Schlüssel `label|function|n|seed` in der CSV erkannt und
übersprungen. Mehr Seeds: Befehl mit größerer Seed-Zahl starten.

**Folge daraus:** Ein Label nie für eine geänderte Konfiguration
wiederverwenden. Sonst wird die neue Konfiguration stillschweigend
übersprungen. Für jede neue Konfiguration ein neues Label vergeben.

### Spalten der Ergebnis-CSV

`group, label, function, n, seed, pop, sigma0, success, f_best, iterations,
evals, budget, final_rank, mean_rank, max_rank, final_sigma, wall_time,
time_per_iter_ms, stop`

## 5. Versuchsaufbau (gilt für alle Stufen)

| Größe | Wert | Begründung |
|---|---|---|
| Dimension | n = 100 | Hauptdimension; Skalierung erst in Stufe 3 |
| Ziel | f < 1e-8 (`fitness_threshold=1e-8`) | Erfolg = bester Wert unter 1e-8 |
| Budget | 10⁴·n = 10⁶ Evaluierungen | umgerechnet in `maximum_iterations = ceil(Budget / Population)` |
| Plateau-Abbruch | aus (`tolerance=0.0`) | sonst stoppen Läufe bei f ≈ 1e-7 vor dem Ziel |
| `sigma_threshold` | 1e-14 | |
| Zeitgrenze | 900 s pro Lauf | nur Sicherheitsnetz |
| Startpunkt | gleichverteilt in den Schranken der Funktion, Generator `default_rng(10000 + seed)` | gleicher Startpunkt für alle Konfigurationen eines Seeds |
| σ0 | 0,3 × Breite des Suchbereichs (meist 3,0) | |
| Schranken, Gradient | nicht verwendet | |
| Seeds | 0 bis 4 | Erweiterung auf 10 möglich |

**Evaluierungen** werden als Iterationen × Population gezählt. Der Solver
wertet pro Iteration zusätzlich den besten Punkt aus; das zählt nicht mit.
Wegen des Aufrundens kann ein gescheiterter Lauf 1 000 008 Evaluierungen
haben.

**Reproduzierbarkeit:** `random_state` allein reicht nicht. Die Startbasis
(`numpy.random.standard_normal`) und zwei numba-Funktionen (`randn`) ziehen
aus globalen Generatoren. `run_single` setzt deshalb vor jedem Lauf
`np.random.seed(seed)` und über eine njit-Funktion den numba-Generator.
Das nicht entfernen.

**Funktionen** (alle aus `seamaze.benchmarks`):

| Funktion | Eigenschaft | Rolle |
|---|---|---|
| Sphere, Bent-Cigar, Discus | achsenparallel, Rang bleibt 1–2 | Kontrolle |
| Ellipsoid | achsenparallel, Rang wächst trotzdem auf ~60 | aussagekräftig |
| Rotated Ellipsoid (f = Σ(Σxᵢ)²) | nicht separabel | aussagekräftig |
| Rosenbrock | nicht separabel, lokales Minimum bei f = 3,9866 | aussagekräftig, aber startpunktabhängig |
| Rastrigin | multimodal | erreicht das Ziel nie; nur Median von f auswerten |

**Kennzahlen:**

- Erfolge k von n Läufen.
- Evaluierungen bis zum Ziel: Median über die erfolgreichen Läufe.
- ERT = Summe der Evaluierungen aller Läufe / Anzahl Erfolge (unendlich ohne
  Erfolg). Bei Rosenbrock unzuverlässig, dort Median + Erfolgszahl nehmen.
- Laufzeit bis zum Ziel: Median über erfolgreiche Läufe. Aus parallelen
  Läufen nur grob vergleichbar.
- Mittlerer Rang über den Lauf, Endrang.
- Vergleich immer relativ zur Standardeinstellung (Label `default`) auf
  derselben Funktion.

Streuung zwischen Seeds (Spanne / Median der Evaluierungen): 2–8 % auf
Sphere, Bent-Cigar, Discus, Ellipsoid, Rotated Ellipsoid; 31 % auf
Rosenbrock. Unterschiede unter etwa 10 % nicht als Effekt deuten.

## 6. Bisherige Ergebnisse

### 6.1 Pilot

Budget von 10⁶ reicht für die Standardeinstellung auf allen sechs
Hauptfunktionen. Teuerste Funktion: Ellipsoid mit etwa 68 % des Budgets.

### 6.2 Stufe 1: je ein Parameter um den Standard variiert

900 Läufe, 808 erfolgreich. Misserfolge: Rastrigin 30/30, Rosenbrock 51/145
(alle im lokalen Minimum f = 3,9866), Ellipsoid 10/145 (alle mit
`update_interval` 10 oder 20), Rotated Ellipsoid 1/145.

Evaluierungen bis zum Ziel relativ zu `default` (Median erfolgreicher Läufe):

| Konfiguration | Ellipsoid | Rotated Ellipsoid | Rosenbrock | Bent-Cigar | Discus |
|---|---|---|---|---|---|
| tol=1e-4 / 1e-3 | 1,11 | 0,98–1,00 | 1,00–1,02 | 1,00 | 1,00 |
| tol=0,05 | 0,85 | 1,09 | 1,19 | 1,00 | 1,00 |
| tol=0,1 | 0,71 | 1,27 | 1,25 | 1,00 | 1,00 |
| tol=0,3 | 0,31 | 1,25 | 1,02 | 1,01 | 1,01 |
| rmax=5 | 0,54 | 1,18 | 1,20 | 1,00 | 1,00 |
| fester Rang 1 | 0,31 | 1,44 | 1,15 | 1,00 | 1,00 |
| fester Rang 2 | 0,30 | 1,38 | 1,03 | 1,00 | 1,00 |
| fester Rang 10 | 0,63 | 1,13 | 1,04 | 1,32 | 1,76 |
| fester Rang 50 | 1,09 | 0,99 | 1,06 | 1,94 | 3,03 |
| interval=1 | 0,41 | 0,87 | 0,70 | 1,00 | 1,00 |
| interval=2 | 0,65 | 1,04 | 0,90 | 1,31 | 1,74 |
| interval=5 | 1,25 | 1,34 | 1,33 | 2,24 | 3,64 |
| interval=10 / 20 | kein Erfolg | 1,43–1,45 | kein Erfolg | 3,7–6,5 | 6,2–10,6 |
| pop=34 / 68 / 136 | 1,09 / 1,22 / 1,31 | 1,38 / 1,99 / 3,05 | 1,23 / 1,63 / 1,74 | 1,37 / 2,07 / 3,39 | 1,05 / 1,15 / 1,34 |

Absolutwerte der Standardeinstellung (Median): Sphere 11,7k, Bent-Cigar 36k,
Discus 86k, Rotated Ellipsoid 326k, Rosenbrock 566k (3 von 5 Erfolge),
Ellipsoid 686k Evaluierungen. Mittlerer Rang: Ellipsoid 59, Rosenbrock 23,
Rotated Ellipsoid 9, sonst 1–1,4.

Rastrigin, Median von f nach Population 17 / 34 / 68 / 136 / 272 / 544:
285 / 135 / 77 / 44 / 21 / 5.

**Schlussfolgerungen aus Stufe 1:**

1. Relevant sind nur zwei Dinge: die **Rangsteuerung** und das
   **Update-Intervall**.
2. `low_rank_energy_tolerance`, `low_rank_max_dimension` und fester Rang
   steuern dasselbe (die Ranggröße). Künftig nur noch eine Achse.
3. Startrang und σ0 haben keinen Effekt. Nicht weiter untersuchen.
4. Größere Population kostet auf unimodalen Funktionen mehr Evaluierungen;
   das gilt für jedes CMA-ES. Nicht weiter untersuchen.
5. Auf dem achsenparallelen Ellipsoid ist ein niedriger Rang 3,3-mal besser,
   auf dem Rotated Ellipsoid schadet er (bis 1,44×). Der adaptive Rang wächst
   auf dem Ellipsoid auf ~60, obwohl Rang 1 reicht. **Ursache ungeklärt.**
6. `update_interval=1` spart Evaluierungen (Ellipsoid 0,41×), macht aber jede
   Iteration etwa 3-mal teurer. Das automatische Intervall kostet also
   Evaluierungen zugunsten der Rechenzeit.
7. 47 % aller Vergleiche lagen innerhalb von 3 % des Standards, weil auf
   Sphere, Bent-Cigar und Discus der Rang bei 1–2 bleibt und die Parameter
   dort nicht wirken können.

### 6.2b Stufe 2: Raster Toleranz × Intervall (abgeschlossen 17.11.2026)

500 Läufe, 5 Seeds, n = 100, keine Abstürze, kein Ruhezustand. Dateien:
`results/stage2/` (stage2.csv, stage2_summary.csv, stage2_table.txt,
plots/evals.png, plots/time.png, plots/rank.png).

Evaluierungen / Laufzeit relativ zum Standard (tol=0,01, Intervall auto),
Median erfolgreicher Läufe; Laufzeiten aus parallelen Läufen (ca.):

| Funktion | bestes in Evaluierungen | bestes in Laufzeit | Kommentar |
|---|---|---|---|
| Ellipsoid | tol=0,3 / tol≤0,1 mit Intervall 1: 0,31–0,41× | tol=0,3: 0,16× Zeit (Rang 2) | Intervall 1 spart Evaluierungen, aber keine Zeit (1,03–1,28×) |
| Rotated Ellipsoid | tol≤0,01, Intervall 1: 0,83–0,87× | tol 0,05–0,3, Intervall 2/3: 0,89–0,95× | Intervall 1 kostet 2,5–3,5× Zeit; höhere tol 1,1–1,3× Evaluierungen |
| Rosenbrock | tol≤0,01, Intervall 1: 0,69–0,70× | tol 0,05–0,3, auto/1: 0,73–0,85× | Erfolg hängt am Seed (Seed 3 nie, Seed 4 halb), nicht an der Einstellung |
| Bent-Cigar, Discus | alle tol gleich (Rang 1) | – | Intervall 2/3 kostet 1,3–2,4× Evaluierungen; auto = 1 bei Rang 1 |

Mittlerer Rang: steigt mit kleiner tol stark (Ellipsoid 2 → 74, Rosenbrock
2 → 54, Rotated Ellipsoid 2 → 32). Großes festes Intervall senkt den Rang
(Rotated Ellipsoid Intervall 3: Rang 3), weil die Rangwahl nur bei Updates
läuft.

Schlussfolgerungen:

1. Kein Paar dominiert überall. Höhere Toleranz hilft dem Ellipsoid stark
   und kostet auf Rotated Ellipsoid/Rosenbrock 1,1–1,3× Evaluierungen.
2. `update_interval=1` lohnt sich nur, wenn Funktionsauswertungen teuer sind
   (bis 0,4× Evaluierungen); in Rechenzeit ist auto meist gleich gut oder
   besser.
3. Feste Intervalle 2/3 sind nie besser als auto in Evaluierungen; auto ist
   eine gute Heuristik, solange der Rang klein bleibt.
4. Kompromiss-Kandidat tol=0,05 (auto): Ellipsoid 0,85× Eval./0,73× Zeit,
   Rotated Ellipsoid 1,09×/1,04×, Rosenbrock 1,19× (verrauscht)/0,94×,
   Bent-Cigar/Discus 1,0×.

### 6.3 Kurztest CMA-ES gegen DLR (n = 30, ein Seed)

| Funktion | CMA-ES | DLR, voller Rang, Standardintervall |
|---|---|---|
| Sphere | 7112 Evaluierungen | 3948 |
| Rotated Ellipsoid | 14 462 | 26 572 |

DLR ist auf der Sphere fast doppelt so schnell wie CMA-ES. **Ursache
ungeklärt**, siehe Abschnitt 8, Schritt 2.

### 6.4 Randnotiz: Rotations-Pilot (nicht Teil der Hauptstudie)

Funktionen f(Qx) mit zufälliger Rotation Q, 2 Seeds, n = 100:

- Rotierter Bent-Cigar: alle Varianten erfolgreich, fester Rang 1 am besten
  (36k gegen 46k Evaluierungen beim Standard).
- Rotierter Discus: keine Variante erfolgreich. Standard f ≈ 160,
  `interval=1` f ≈ 1e-4, niedrige Ränge f ≈ 400–2100.
- Rotierter Ellipsoid: nur `interval=1` erfolgreich (753k Evaluierungen).
  Standard f ≈ 9, niedrige Ränge f ≈ 300–2700.

Aussage: Rangempfehlungen aus achsenparallelen Funktionen nicht
verallgemeinern.

## 7. Stufenleiter (`ladder`): abgeschlossen und ausgewertet

**Status:** 305 Läufe fertig, keine Abstürze. Auswertung mit
`analyze_ladder.py` (Schritt 1 aus Abschnitt 8 ist damit erledigt). Ausgaben:
`results/ladder_table.txt`, `results/ladder_plots/ladder.png`,
`results/ladder_plots/fixed_rank.png`.

Evaluierungen bis zum Ziel (Median erfolgreicher Läufe, n = 100, 5 Seeds;
Stufen 1 bis 4 mit Intervall 1):

| Funktion | 1 CMA-ES | 2 voller Rang | 3 bester fester Rang | 4 adaptiv | 5 Standard |
|---|---|---|---|---|---|
| Sphere | 40,2k | 11,5k | 11,7k (r=1) | 11,7k | 11,7k |
| Bent-Cigar | 92,6k | 35,9k | 35,8k (r=1) | 35,9k | 35,9k |
| Discus | 97,0k | 86,0k | 85,8k (r=1) | 85,5k | 85,5k |
| Ellipsoid | 301k | 207k | 211k (r=1) | 285k | 686k |
| Rotated Ellipsoid | 69,4k | 218k | 243k (r=50) | 275k | 318k |
| Rosenbrock | 426k | 395k | 392k (r=50) | 399k (3/5) | 566k (3/5) |

Faktor zur Stufe davor:

| Funktion | 2/1 Formulierung | 3/2 Niedrigrang | 4/3 Rangwahl | 5/4 Intervall |
|---|---|---|---|---|
| Sphere | 0,29 | 1,01 | 1,01 | 1,00 |
| Bent-Cigar | 0,39 | 1,00 | 1,00 | 1,00 |
| Discus | 0,89 | 0,99 | 1,00 | 1,00 |
| Ellipsoid | 0,69 | 1,01 | 1,36 | 2,41 |
| Rotated Ellipsoid | 3,14 | 1,12 | 1,13 | 1,16 |
| Rosenbrock | 0,93 | 0,99 | 1,02 | 1,42 |

Fester Rang mit Intervall 1 (Evaluierungen): Rotated Ellipsoid fällt monoton
von 459k (r=1) auf 218k (r=100); Rosenbrock von 650k (r=1, 3/5) auf 395k;
Ellipsoid 211k (r=1), 293k (r=25), 288k (r=50), 207k (r=100), also in der
Mitte am schlechtesten.

Mit Auto-Intervall: `CMA-ES` scheitert auf dem Ellipsoid (0/5, f ≈ 2e-8 am
Budgetende) und braucht auf dem Discus 374k statt 97k. `full rank` braucht
auf dem Discus 360k statt 86k. Rastrigin, Median f: CMA-ES 262, DLR 285.

**Zuordnung:**

- (c) CMA-ES selbst: Rastrigin (beide scheitern gleich). Der Verlust durch
  große Update-Intervalle tritt in CMA-ES genauso auf.
- DLR-Formulierung (nicht der Rang): Rotated Ellipsoid, 3,1-mal schlechter
  als CMA-ES schon bei vollem Rang. Größter Einzelverlust. Ursache offen.
- (a) Niedrigrang: Rotated Ellipsoid und Rosenbrock werden mit kleinerem Rang
  stetig schlechter (r=1: 2,1× bzw. 1,65× gegenüber vollem Rang).
- (b) Rangwahl: Ellipsoid 1,36×, sonst ≤ 1,13×. In Evaluierungen klein, in
  Rechenzeit groß (Ellipsoid 160 s adaptiv gegen 16 s mit Rang 1).
- (b) Intervall-Automatik: Ellipsoid 2,41×, Rosenbrock 1,42×.

**Nachtrag: Die CMA-ES-Referenz der Bibliothek hat einen Fehler im
Schrittweiten-Pfad (geklärt, ein Seed, n = 100).**

In `seamaze/optimizers/_cmaes.py`, Funktion `_tell`, wird der Pfad p_σ mit
D⁻¹·Bᵀ·y aktualisiert statt mit B·D⁻¹·Bᵀ·y. Der Pfad liegt damit in den
Koordinaten der Eigenbasis B. Ändert sich B zwischen zwei Updates (bei fast
gleichen Eigenwerten, etwa auf der Sphere, liefert die Eigenzerlegung jedes
Mal eine andere Basis), passen aufeinanderfolgende Beiträge nicht mehr
zusammen, der Pfad wird zu lang (|p_σ|/E = 0,90 statt 0,69) und σ sinkt zu
langsam. DLR hat das Problem nicht, weil es C^(-1/2)·y per Lanczos im
Originalraum berechnet.

Test mit `cmaes_path_variant.py` (Kopie mit genau dieser einen Korrektur,
erzeugt durch `make_variant.py`; Vergleich mit `diag_path.py`),
Evaluierungen Bibliothek → Variante, Intervall 1:
Sphere 40,2k → 11,7k; Bent-Cigar 93,6k → 36,4k; Discus 95,9k → 89,2k;
Ellipsoid 302k → 306k; Rotated Ellipsoid 69,3k → 65,1k; Rosenbrock 425k →
397k.

Folgen:

- Der scheinbare Vorsprung von DLR auf Sphere und Bent-Cigar (Faktor 0,29
  und 0,39 in der Tabelle oben) ist ein Artefakt der Referenz. Mit der
  Variante sind beide Verfahren dort gleich.
- Echt bleiben: Ellipsoid (DLR voller Rang 207k gegen 306k) und Rotated
  Ellipsoid (DLR 218k gegen 65k).
- Für alle weiteren Vergleiche als Stufe 1 die Variante verwenden, nicht
  `CMAES` aus der Bibliothek.
- Widerlegt: Die Vermutung, das ψ-Update wirke in DLR als zweite
  Schrittweitensteuerung. Mit eingefrorener Kovarianz brauchen CMA-ES
  (11,5k) und DLR (12,1k) auf der Sphere gleich viel wie DLR Standard
  (11,7k). Skripte: `diag_scale.py`, `diag_weights.py` (negative Gewichte
  sind nicht die Ursache: 40,2k mit, 42,5k ohne).
- Die Bibliothek wurde nicht geändert. Ob der Fehler den Autoren gemeldet
  wird, entscheidet der Nutzer.

**Nachtrag 2 (17.11.2026): korrigierte Referenz und Ursache Rotated
Ellipsoid.**

- `ladder_ref` (Labels `CMA-ES korr.`, `CMA-ES korr., interval=1`, 5 Seeds)
  ist gelaufen; `analyze_ladder.py` nutzt jetzt diese Referenz. Plots:
  `ladder_plots/ladder_korr.png`, `fixed_rank_korr.png` (alte Dateinamen
  waren gesperrt). Faktor voller Rang / CMA-ES korr.: Sphere 0,97,
  Bent-Cigar 0,98, Discus 0,97, Ellipsoid 0,69, Rotated Ellipsoid 3,31,
  Rosenbrock 1,01.
- Auch CMA-ES korr. mit eigenem Auto-Intervall (6) verliert stark (Ellipsoid
  2/5 Erfolge, Discus 361k statt 89k): großes Intervall schadet CMA-ES
  allgemein, nicht nur DLR.
- Fehlerquelle CMA-ES-Bibliothek: Hansen-Tutorial (arXiv:1604.00772),
  Gl. (31) S. 20 und Erklärung S. 21: C^(-1/2) = B·D⁻¹·Bᵀ, das vordere B
  fehlt in `_cmaes.py` Zeile 807-808.
- `diag_rotell.py`: Lanczos ist genau (Fehler ~1e-5). DLR lernt auf dem
  Rotated Ellipsoid die Kovarianz kaum (cond(A^½CA^½) bleibt ~1,3e4, CMA-ES
  sinkt auf 116). min ψ / λ_min(C) ≈ 0,9: ψ bildet die Untergrenze.
- Ursache: Im Update (`_adaptive_bug_step`, `ext_s`) geht nur U·S·Uᵀ ein,
  ψ zerfällt nur passiv. Test `dlr_variant_psifold.py` (ψ ins Update
  projiziert), Seed 0: voller Rang Rotated Ellipsoid 225k → 64,6k (= CMA-ES),
  Ellipsoid 207k → 299k (Vorteil weg). Mit adaptivem Rang ist die Variante
  katastrophal (Sphere 942k statt 12k, Ellipsoid/Rosenbrock scheitern), weil
  Varianz außerhalb des Unterraums verloren geht. Ergebnis:
  struktureller Zielkonflikt der Darstellung Niedrigrang + Diagonale,
  Kategorie (a). Keine einfache Korrektur.
- Ungültig wegen Ruhezustand des Rechners (13:38-14:34): in
  `results/diag/psifold_s0.txt` die Zeilen „Ellipsoid | voller Rang |
  Bibliothek" und „Rotated Ellipsoid | adaptiv | psifold". Vor langen Läufen
  Ruhezustand ausschalten lassen.

**Nachtrag 3 (17.11.2026): ψ-Befund mit 5 Seeds bestätigt, Rangwahl
untersucht.**

- `run_study.py psifold 5 5` → `results/psifold.csv` (Label `psifold full
  rank, interval=1`). Median Evaluierungen, psifold / DLR Bibliothek / CMA-ES
  korr. (alle voller Rang bzw. Intervall 1): Sphere 11,8k / 11,5k / 11,8k;
  Bent-Cigar 36,3k / 35,9k / 36,4k; Discus 87,8k / 86,0k / 88,5k; Ellipsoid
  294k / 207k / 300k; Rotated Ellipsoid 64,7k / 218k / 65,8k; Rosenbrock
  398k / 395k / 393k. Mit ψ im Update ist DLR bei vollem Rang auf allen
  Funktionen gleich CMA-ES. Einziger Unterschied der Bibliotheks-DLR bei
  vollem Rang ist also die ψ-Behandlung.
- `diag_rank.py` (Unterklasse mit unveränderter `tell` plus Protokoll; auf
  der Sphere identisch zur Bibliothek geprüft). Ellipsoid, Seed 0, adaptiv,
  Intervall 1: Rang steigt auf 86, C ist stark korreliert (max |Korrelation|
  bis 0,97, ‖Korr − I‖_F bis 16), obwohl die optimale Kovarianz diagonal ist.
  Rang-Vorschläge 2. Laufhälfte (Schwelle × Toleranz): 0,02/0,01 → 84;
  0,2/0,01 → 42; 0,2/0,3 → 16. Kein vernünftiger Schwellwert liefert Rang 1.
- Identitätsprüfung: Die instrumentierten Läufe reproduzieren exakt die
  Bibliotheksläufe `interval=1`, Seed 0, aus Stufe 1 (Sphere 11 696,
  Ellipsoid 284 631, Rotated Ellipsoid 274 108 Evaluierungen). Die
  kontrafaktischen Vorschläge nutzen ab dem 18. Augmentierungsvektor andere
  Zufallsvektoren als der numba-Solver und sind daher nur Näherungen.
- `diag_corr.py` (Kontrolle, nur Seed 0): Das korrigierte CMA-ES baut auf dem
  Ellipsoid dieselben Korrelationen auf (max 0,94–0,98, ‖·‖_F bis 18, gegen
  Ende 3,6). Die Korrelationen sind echte, vorübergehende Struktur der
  Kovarianz-Adaption (Annäherungsphase), kein DLR-Artefakt.
- Der Rang bleibt hoch, obwohl die Korrelationen gegen Ende auf das Niveau
  des CMA-ES abklingen (‖·‖_F 16 → 3,7; am Ende überleben noch 56
  Eigenwerte die Schwelle 0,2). Ursache ist nicht die Schrittweite ±1,
  sondern das Kriterium selbst: Es misst nur, wie sich die Energie auf
  Richtungen verteilt. Schwache, auf viele Richtungen verteilte Struktur
  verlangt viele Ränge. Keine Einstellung von Toleranz oder Schwelle liefert
  Rang 1 (alle Vorschläge ≥ 16), also kein Abstimmungsproblem.
- Fester Rang auf dem Ellipsoid (Seed 0, `diag_midrank.py`): r=1 211k,
  r=25 293k, voller Rang 206k, CMA-ES korr. 300k. Diagonal-Fehler (Std von
  log10(diag(C)·coef)) ist bei r=25 nicht schlechter (Ende 0,17 gegen 0,35
  bei r=1). Die Vermutung „ψ wird verdrängt" ist damit nicht gestützt. Bei
  r=25 bleibt dagegen cond(A^½CA^½) länger hoch (7,8e2 bei 241k; r=1 und
  r=100 fallen am Ende auf 20–35): Mittlere Ränge tragen die vorübergehenden
  Korrelationen länger mit, wie CMA-ES (gleiche Evaluierungszahl). Rang 1
  kann sie nicht darstellen und ist deshalb schneller. Warum der volle Rang
  ebenfalls schnell ist, ist offen (ψ-Anteil springt am Ende von 2 % auf
  81 %).
- Zuordnung Rangwahl: Die Heuristik arbeitet wie entworfen, ihr Kriterium
  unterscheidet aber nicht zwischen nützlicher und nutzloser Struktur
  (r=1 ≈ voller Rang auf dem Ellipsoid). Das ist eine Begrenzung der
  Heuristik, Kategorie (b). Die Mehr-Evaluierungen entstehen durch mittlere
  Ränge, die vorübergehende Korrelationen mittragen; das betrifft CMA-ES
  ebenso. Alles nur Seed 0.

**Offen (nächste Schritte in dieser Reihenfolge; Punkt 1 und 2 sind durch
die Nachträge erledigt):**

1. Warum ist DLR bei vollem Rang auf dem Rotated Ellipsoid 3,1-mal
   schlechter als CMA-ES? Kandidaten (alles Vermutungen): Lanczos-Näherung
   von C^(-1/2) mit nur 20 Krylov-Vektoren, Umverteilung zwischen S und ψ
   (Faktor `alpha` am Ende von `_adaptive_bug_step`), Integrator.
2. Warum ist DLR auf Sphere und Bent-Cigar 2,6- bis 3,5-mal besser als
   CMA-ES? Die Schrittweiten-Konstanten beider Solver sind identisch
   (geprüft: `_lr_sigma`, `_damp_sigma`, `_mu_eff`, Lernraten). Die Ursache
   muss im Kovarianz-Update liegen. Verlauf von σ vergleichen (Schritt 2 in
   Abschnitt 8).
3. Danach Schritt 3 bis 6 aus Abschnitt 8.

Ursprünglicher Startbefehl und Aufbau:

```
../../.venv/Scripts/python.exe run_study.py ladder 6 5 > results_ladder.log 2>&1
```

305 Läufe, geschätzt 1,5 bis 2 Stunden. Fortschritt: letzte Zeile von
`results_ladder.log`. Falls der Lauf abgebrochen wurde: denselben Befehl
erneut starten.

**Idee:** Fünf Stufen vom normalen CMA-ES bis zur Standardeinstellung. Jede
Stufe ändert genau eine Sache. Der Sprung zwischen zwei Stufen zeigt, was
diese eine Sache kostet.

| Stufe | Konfiguration | Label | CSV | Sprung zur Stufe davor bedeutet |
|---|---|---|---|---|
| 1 | CMA-ES, Intervall 1 | `CMA-ES, interval=1` | ladder | Referenz. Schlecht hier → Kategorie (c) |
| 2 | DLR, Rang n, Intervall 1 | `full rank, interval=1` | ladder | Kosten der DLR-Formulierung ohne Rangbeschränkung → (a) im weiteren Sinn |
| 3 | DLR, bester fester Rang, Intervall 1 | `fixed r=K, interval=1`, K ∈ {1, 2, 5, 10, 25, 50} | ladder | Effekt der Niedrigrang-Näherung → (a) |
| 4 | DLR, adaptiver Rang, Intervall 1 | `interval=1` | stage1 | Güte der Rangwahl → (b) |
| 5 | DLR Standard | `default` | stage1 | Effekt des automatischen Intervalls → (b) |

Zusätzlich in `ladder.csv`: `CMA-ES` (eigenes Standardintervall, auch auf
Rastrigin) und `full rank` (DLR-Automatik-Intervall). Bei n = 100 und Rang
100 ergibt die DLR-Automatik ebenfalls 6, also direkt mit `CMA-ES`
vergleichbar. In `stage1.csv` liegen außerdem `fixed r=K` mit
Automatik-Intervall.

## 8. Nächste Schritte

### Schritt 1: Stufenleiter auswerten

Neues Skript `analyze_ladder.py` schreiben. `analyze.py` dafür nicht umbauen.

1. `results/stage1.csv` und `results/ladder.csv` einlesen und untereinander
   hängen. Aus `stage1.csv` nur die Labels `default`, `interval=1` und
   `fixed r=K` behalten.
2. Pro (Label, Funktion): Erfolge k/n, Median der Evaluierungen der
   erfolgreichen Läufe, Median der Laufzeit, mittlerer Rang.
3. Pro Funktion für Stufe 3 den festen Rang K mit den wenigsten
   Evaluierungen bestimmen („bester fester Rang"). Nur Ränge mit 5/5
   Erfolgen zulassen; wenn es keinen gibt, den mit den meisten Erfolgen.
4. Tabelle ausgeben: Zeilen = Funktionen, Spalten = Stufen 1 bis 5, Werte =
   Evaluierungen absolut und als Faktor zu Stufe 1. Als Datei
   `results/ladder_table.txt` speichern.
5. Plot nach Abschnitt 9 erzeugen.
6. Für Rastrigin nur Median von f für `CMA-ES` und `default` vergleichen.

**Zuordnungsregel** (Faktor = Evaluierungen der Stufe / Evaluierungen der
Stufe davor; Schwelle 1,15 wegen der Seed-Streuung):

| Beobachtung | Zuordnung |
|---|---|
| Stufe 1 erreicht das Ziel nicht oder braucht fast das ganze Budget | (c) CMA-ES selbst |
| Stufe 2 / Stufe 1 > 1,15 | DLR-Formulierung (Integrator, Aufteilung in U·S·Uᵀ und ψ), nicht der Rang |
| Bester fester Rang / Stufe 2 > 1,15 | (a) Niedrigrang-Näherung kostet |
| Bester fester Rang / Stufe 2 < 0,87 | Niedrigrang hilft sogar |
| Stufe 4 / bester fester Rang > 1,15 | (b) Rangwahl-Heuristik |
| Stufe 5 / Stufe 4 > 1,15 | (b) Intervall-Heuristik |

Rosenbrock: wegen der Startpunktabhängigkeit zusätzlich die Erfolgszahlen
nennen und keine Faktoren unter 1,3 deuten.

Dem Nutzer berichten: eine Tabelle Funktion × Stufe, dann pro Funktion ein
Satz zur Zuordnung.

### Schritt 2: Sphere-Auffälligkeit klären

Frage: Warum braucht DLR auf der Sphere weniger Evaluierungen als CMA-ES
(Abschnitt 6.3)? Erst prüfen, ob es bei n = 100 und 5 Seeds bestehen bleibt
(Labels `CMA-ES`, `CMA-ES, interval=1`, `full rank`, `default` auf Sphere in
den CSVs). Wenn ja:

- In beiden Solvern Lernrate und Dämpfung der Schrittweite vergleichen
  (`_lr_sigma`, `_damp_sigma`, `_expected_path_length`, `_mu_eff`,
  Gewichte). Werte für n = 100 ausgeben und nebeneinanderstellen.
- Verlauf von σ über die Iterationen für einen Seed vergleichen
  (Callback, der `solver._sigma` mitschreibt).
- Ergebnis als Befund berichten, nichts am Code ändern.

### Schritt 3: Wechselwirkung Rangsteuerung × Intervall (Stufe 2)

**Status 17.11.2026: umgesetzt und gestartet** (`run_study.py stage2 6 5`,
500 Läufe, Log `results/logs/stage2.log`). Konfigurationen in
`study.stage2_configs`, Auswertung `analysis/analyze_stage2.py` (Heatmaps
Evaluierungen, Laufzeit, Rang; Tabelle mit bester Einstellung je Funktion).
Nach Abschluss: Auswertung laufen lassen, Plots ansehen, Ergebnisse in
Abschnitt 6 eintragen, dann Rosenbrock auf 10 Seeds erweitern (z. B. eigene
Stufe mit nur Rosenbrock oder `stage2 6 10` und die übrigen Funktionen
vorher in `STAGE2_FUNCTIONS` auskommentieren; dabei das Label-Schema nicht
ändern).

- Raster: `low_rank_energy_tolerance` ∈ {1e-3, 1e-2, 5e-2, 1e-1, 3e-1} ×
  `update_interval` ∈ {automatisch, 1, 2, 3} = 20 Konfigurationen.
- In `study.py` eine Funktion `stage2_configs` ergänzen, Gruppe `stage2`,
  Labels der Form `tol=0.05, interval=2` bzw. `tol=0.05, interval=auto`.
  Bei „automatisch" `update_interval` nicht in die kwargs aufnehmen.
- Alle 20 Konfigurationen neu rechnen (auch die, die es in Stufe 1 schon
  gibt). Das ist einfacher als das Zusammenführen und kostet nur etwa
  40 Minuten mehr.
- Funktionen: Ellipsoid, Rotated Ellipsoid, Rosenbrock, dazu Bent-Cigar und
  Discus als Kontrolle. Sphere weglassen.
- Seeds: 5. Für Rosenbrock danach auf 10 erweitern.
- In `run_study.py` die Stufe `stage2` in `build_tasks` ergänzen.
- Geschätzte Dauer: 2 bis 2,5 Stunden.

### Schritt 4: Skalierung über die Dimension (Stufe 3)

- Dimensionen: n = 30, 100, 300. n = 1000 nur für Sphere und Bent-Cigar
  und nur mit einem Seed.
- Konfigurationen: `default`, `interval=1`, `CMA-ES`, dazu höchstens zwei
  auffällige aus Schritt 1 oder 3.
- Funktionen: Ellipsoid, Rotated Ellipsoid, Rosenbrock, Bent-Cigar.
- `build_tasks` nutzt bisher die Konstante `NDIM`. Für diese Stufe die
  Dimension als Schleife in die Aufgabenliste aufnehmen (das Tupel hat das
  Feld `n` schon).
- Budget bleibt 10⁴·n. Die Zeitgrenze `WALL_CAP` für n = 300 auf 3600 s
  erhöhen.
- **Laufzeit sauber messen:** einen Seed pro Konfiguration mit nur einem
  Prozess laufen lassen (`run_study.py <stufe> 1 1`) und nur diese Zeiten
  für Laufzeitaussagen verwenden.
- Vorher mit einem einzelnen Lauf bei n = 300 die Dauer prüfen und dem
  Nutzer nennen.

### Schritt 5 (nur nach Rückfrage): interne Konstanten

**Teilweise erledigt:** Der Filter-Schwellwert `threshold` ist durch die
kontrafaktischen Rang-Vorschläge in `diag_rank.py` beantwortet (Nachtrag 3:
keine Schwelle bis 0,2 liefert auf dem Ellipsoid Rang 1). Nicht erneut
rechnen. Offen bleibt nur `beta`.

Wenn Schritt 1 die Rangwahl als Ursache zeigt (Ellipsoid: Rang ~60 statt 1):

- `seamaze/optimizers/_dlrcmaes.py` nach `studies/parameter_study/variants/dlr_variant.py`
  kopieren. In der Kopie `threshold` (Standard 0,02) und `beta` (Standard 0)
  als Argumente durchreichen. Das Original nicht ändern.
- Prüfen, dass die Kopie mit Standardwerten dieselben Ergebnisse liefert wie
  das Original (gleicher Seed, gleiche Iterationszahl).
- Variieren: `threshold` ∈ {0,005, 0,02, 0,05, 0,1}, `beta` ∈ {0, 0,25, 0,5,
  1}, jeweils einzeln.

### Schritt 6: Abschlussbericht

Für den Nutzer, auf Deutsch, in dieser Reihenfolge:

1. Zuordnungstabelle: Funktion × Ursache (a), (b), (c).
2. Die zwei bis drei wichtigsten Befunde mit Zahlen.
3. Was offen bleibt und warum.
4. Pfade der Plots und Tabellen.

## 9. Vorgaben für Plots

Der Nutzer war mit den bisherigen Plots nicht zufrieden. Bekannte Mängel, die
nicht wiederholt werden dürfen:

- Linien oder Panels, die nichts zeigen (alle Werte bei 1,0; mittlerer Rang
  bei festem Rang; Erfolgsquote, wenn fast alles erfolgreich ist).
- Leere Positionen auf der x-Achse.
- Achsenbeschriftungen in der Form „6 × 10⁻¹". Stattdessen feste Marken wie
  0,25 / 0,5 / 1 / 2 / 4 setzen.
- Ein Plot pro Parameter, wenn mehrere Parameter dasselbe zeigen.

Grundsatz: **ein Plot pro Aussage.** Vor dem Erstellen den Satz aufschreiben,
den der Plot belegen soll, und ihn als Titel verwenden.

Vorgesehene Plots:

1. **Stufenleiter** (`results/ladder_plots/ladder.png`): ein Panel pro
   Funktion (Ellipsoid, Rotated Ellipsoid, Rosenbrock, Bent-Cigar, Discus,
   Sphere), x-Achse = Stufen 1 bis 5, y-Achse = Evaluierungen bis zum Ziel
   (log, absolute Zahlen). Balken oder Punkte mit den Einzelläufen als kleine
   Punkte darüber. Stufe 3 mit dem gewählten Rang beschriften („r = 2").
   Stufen ohne vollen Erfolg mit „k/5" beschriften, ohne Erfolg grau.
2. **Fester Rang** (`ladder_plots/fixed_rank.png`): x-Achse = Rang (1, 2, 5,
   10, 25, 50, 100), y-Achse = Evaluierungen, eine Linie pro Funktion, nur
   Intervall 1. Waagerechte Marke für den adaptiven Rang (`interval=1`) je
   Funktion. Nur Funktionen aufnehmen, bei denen sich etwas ändert.
3. **Raster aus Schritt 3:** je Funktion eine Heatmap Toleranz × Intervall,
   Werte relativ zum Standard, Zahl in jeder Zelle.
4. **Skalierung aus Schritt 4:** Evaluierungen und sequenziell gemessene
   Laufzeit über n, eine Linie pro Konfiguration, ein Panel pro Funktion.

Technik: matplotlib mit `matplotlib.use('Agg')`, 150 dpi. Farben in fester
Reihenfolge je Funktion wie in `analyze.py` (`COLORS`, `FUNC_ORDER`),
zusätzlich unterschiedliche Marker. Gitter und Achsen dezent, keine oberen
und rechten Rahmenlinien. Jeden Plot nach dem Erzeugen ansehen und auf
überlappende Beschriftungen prüfen.

## 10. Bekannte Fallstricke

- **Startverzögerung:** Jeder Prozess kompiliert beim Start die
  numba-Funktionen. Die ersten Ergebnisse kommen erst nach 1 bis 3 Minuten,
  die Logdatei bleibt so lange leer. Das ist kein Fehler.
- **Reihenfolge:** Teure Funktionen laufen zuerst (Ellipsoid, Rosenbrock,
  Rastrigin, Rotated Ellipsoid). Der Fortschritt wirkt anfangs langsam.
- **Laufzeiten:** Zeiten aus Läufen mit 6 Prozessen nur grob vergleichen.
- **Shell:** Mehrzeilige Python-Skripte nicht per Heredoc an die Bash
  übergeben, wenn sie Apostrophe in Kommentaren oder Docstrings enthalten.
  Stattdessen als Datei schreiben und ausführen.
- **`analyze.py`:** Die Reihenfolge der Labels kommt aus `stage1_configs`.
  Unbekannte Labels werden alphabetisch angehängt.
- **Rosenbrock:** Seed 3 und 4 landen in etwa 60 % der Läufe im lokalen
  Minimum, Seed 0 bis 2 in 15–25 %. ERT dort nicht verwenden.
- **CMA-ES:** kann zusätzlich mit `NO_EFFECT_AXIS` abbrechen. Die Spalte
  `stop` prüfen, bevor ein Misserfolg gedeutet wird.
- **Befunde und Vermutungen trennen.** Ursachen, die nicht durch einen
  eigenen Lauf belegt sind, im Bericht ausdrücklich als Vermutung
  kennzeichnen.
