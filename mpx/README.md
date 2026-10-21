# MPX ROS node

**Model update, 2026-10-07:** TITA physical model contents in `tita_description`
(xacro, URDF and MuJoCo XML) and standalone `mpx/data/tita/tita.xml` are aligned
with the robot export `bags/tita_running.urdf`: total declared mass 27.68978 kg,
full inertia tensors, joint geometry and URDF limits. ROS MPC/WBC, state estimator,
robot_state_publisher and the robot controlled by `rl_0` share the expanded
`tita_description` model. The RL policy weights, observation/action scaling,
controller torque caps and workspace hardware/simulator plugins remain as configured.
The alignment does not identify or change the factory MCU locomotion policy.
Existing filenames are retained. Rebuild with `colcon build`, source
`install/setup.bash`, and restart running nodes to load the updated model.


**Aggiornamento 26 settembre — causa importante della deriva individuata:**
con il budget predefinito del solver ODE, velocità articolari e posizioni erano
fortemente incoerenti. Il world ora imposta **300 iterazioni QuickStep**, con
fisica a 500 Hz, MPC a 50 Hz, WBC a 100 Hz e LLC a 500 Hz.
Nel confronto GUI da 60 s la deriva massima scende da **32.5 cm a 2.6 mm**.
La verifica lunga GUI di 179.35 s conferma **5.50 mm** di deriva massima,
nessun NaN e nessuna caduta. La verifica finale dopo refactoring e correzione
dello yaw rate misura **5.62 mm in 179.73 s**, controllo medio **99.85 Hz**.
Tracking con GUI: **0.998 m/s** a vx=1.0 e **0.400 rad/s** a wz=0.4. I risultati e
problemi precedenti sono mantenuti come cronologia, non come configurazione attuale.

Dopo l'aggiornamento:

```bash
colcon build --packages-select gazebo_bridge mpx --symlink-install
source install/setup.bash
```

Riavviare Gazebo e MPX: il world e il codice Python già caricati non cambiano
nei processi in esecuzione.

Start the simulation before the controller. The controller waits up to 30 seconds
(wall time, also when Gazebo is paused) for `/tita/robot_description`, using
transient-local QoS so it can receive the existing description after startup.

The node imports that expanded URDF into MuJoCo and passes the same model to the
MPX wrapper. It does not read `mpx/data/tita/tita.xml`. The standalone MPX examples
can still use that file. Restart MPX after changing/restarting the robot model.

The adapter preserves URDF joint transforms, limits, collision shapes, masses and
inertias. It adds a floating base, names collision geometries, and removes visuals
and ROS/Gazebo extensions. Contact simulation remains in Gazebo: MPX uses the
collision shapes for kinematics and its WBC computes the contact forces, so
MuJoCo collision pairs are disabled. This shares the robot description; it does
not make the two physics engines identical. Gravity and controller tuning remain
MPX configuration values. Initial joints, total mass, wheel radius and nominal
wheel spacing are derived from the received model.

Run Gazebo paused, then launch MPX separately:

```bash
ros2 launch sim_bringup sim_bringup.launch.py sim_env:=gazebo
# Another terminal, with the same ROS workspace/environment:
ros2 launch mpx mpx.launch.py
```

Wait for `MPX ready on CPU`, then press Play. `gui:=false` runs Gazebo headless.
Parameters are in `config/mpx_node.yaml`; MPX publishes torque whenever it is active.

The node forces JAX to CPU before importing numerical libraries. It computes
MPC + WBC on one worker every 10 ms of **JointState simulation time**; input
callbacks continue receiving the latest feedback during computation. The
500 Hz effort controller/bridge keeps applying the last valid torque between
updates. No extra 500 Hz Python publisher is needed. Pausing simulation stops
new solves; computation overruns skip old periods and are reported in the log.
100 Hz is the requested rate, not a hard real-time guarantee on this machine.

Optional ROS parameters:

- `robot_description_topic` (default `/tita/robot_description`)
- `robot_description_timeout` (default `30.0` wall seconds)
- `precompile` (default `true`; keep enabled for simulation startup)
- `model_name` (default `tita`)
- `model_states_topic` (default `/model_states`)
- `joint_states_topic` (default `/tita/joint_states`)
- `effort_command_topic` (default `/tita/effort_controller/commands`)

The adapter currently requires the eight TITA joints in controller order, root
`base_link`, two cylindrical wheels of equal radius, and initial joint positions
in `ros2_control`. An incompatible description fails explicitly, without falling
back to a separate XML.

Build: `colcon build --packages-select mpx gazebo_bridge --symlink-install`.
Run with the MPX Python dependencies installed in the active environment.
After sourcing ROS and the workspace, run model checks with:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python3 -m unittest discover -s src/mpx/tests -v
```

MuJoCo URDF import/compiler options:
https://mujoco.readthedocs.io/en/3.3.1/XMLreference.html#compiler

CPU timing and numerical parity between fused and unfused ROS control:

```bash
python3 src/mpx/tests/benchmark_control.py
```

On a **test** Gazebo instance already paused with MPX ready, run the integration
check (it changes pause state and leaves Gazebo paused):

```bash
python3 src/mpx/tests/check_gazebo_feedback.py --seconds 30
```

It checks standing height, finite torque, applied/held effort and pause/resume.
Results default to `/tmp/mpx-feedback-check.json`. See
[MPX_INTEGRATION_REPORT.md](MPX_INTEGRATION_REPORT.md) for measured timings,
previous integration problems, corrections and remaining limitations.

Gazebo GUI defaults to **20 FPS** and **nice 10**, keeping CPU priority for
physics/control. Both are configurable:

```bash
colcon build --packages-select gazebo_bridge sim_bringup --symlink-install
source install/setup.bash
ros2 launch sim_bringup sim_bringup.launch.py sim_env:=gazebo gui_fps:=20 gui_nice:=10
```

This adds a C++ GUI plugin, so build it before using the updated world. The FPS
limit only affects rendering; physics remains 500 Hz. The 30-second GUI test
kept the robot standing at about 75 Hz effective control on this machine; it
does not establish a constant 100 Hz rate. `gui_nice:=0` restores normal client
priority. MPX numerical computation remains CPU-only.

The ROS WBC now receives the commanded COM height (currently 0.4 m) and zero
vertical velocity as actual task references, while retaining MPC acceleration
feedforward. This prevents the slow extension drift observed in the longer GUI
test. Standalone wrapper callers retain the old behavior unless they supply
`com_height_target`.

The updated configuration passed **118.6 simulated seconds with GUI open**,
including pause/resume: maximum base tilt 0.49°, no invalid WBC results.
It still drifted up to 0.75 m in XY and ran at about 76.6 Hz effective control;
this demonstrates balance, not fixed-position holding or constant 100 Hz.
Detailed measurements and the preceding failed test are in `test_results/`
and the integration report. To repeat after starting the paused simulation
with MPX ready:

```bash
python3 src/mpx/tests/check_gazebo_feedback.py --seconds 120
```


### Analisi della deriva (in corso, 26 settembre)

- **Dati:** base da `/model_states`, giunti da `/tita/joint_states`, modello
  dall'URDF ROS. Test con base inclinata e giunti permutati verificano frame,
  ordine e velocità di COM/ruote. I due topic non sono sincronizzati esattamente:
  `ModelStates` non contiene timestamp. Questo limite resta aperto.
- **Bug corretto:** il riferimento WBC della ruota destra usava la quota della
  sinistra. Il test asimmetrico riproduceva 2 cm di errore; ora passa.
- **Deriva:** i pesi di posizione XY sono zero; il robot non mantiene una
  posizione assoluta. Questo spiega perché non recupera lo spostamento, ma non
  dimostra la causa della velocità residua né della successiva instabilità.
- **Verifica precedente fallita:** una nuova prova GUI con osservatore aggiuntivo
  ha prodotto una caduta e circa 55–62 Hz di elaborazione. Il precedente successo
  di 118.6 s non è una garanzia di stabilità in tutte le condizioni di carico.
  Non è ancora separato l'effetto del carico da quello della correzione WBC.
- **Prossima misura:** traccia a 20 Hz di base, giunti e torque, salvata dal probe
  in `*.trace.json`; analisi cinematica offline per evitare un secondo processo
  MuJoCo/JAX durante il controllo. Confrontare quota, limiti articolari, velocità
  e ritardi prima della caduta, senza dedurre un integratore dalla sola deriva.

- **Diagnostica corretta:** nella versione con `/mpx/estimate/*`, l'attributo
  publisher `_rpy` nascondeva il metodo omonimo per calcolare roll/pitch/yaw.
  L'eccezione veniva ignorata: COM e velocità COM uscivano, orientamento e ruote
  no. Rinominato il publisher e aggiunto test dei cinque messaggi. Questo bug
  riguardava la visualizzazione delle stime, non gli ingressi del solver.
- **Memoria del controller:** non c'è un integratore di errore di posizione nel
  nodo. La posa world viene riletta da Gazebo; le posizioni articolari dal
  broadcaster. Persistono il warm start dell'orizzonte MPC e lo yaw svolto.
  Pertanto non attribuiamo la deriva a un integratore senza evidenza.
- **Meccanismo di peggioramento da verificare:** nel primo stato invalido della
  prova precedente il ginocchio destro aveva circa 0.0086 rad di margine al
  limite e velocità +9.56 rad/s. La previsione libera a 10 ms oltrepassa il
  limite di circa 0.087 rad: la disuguaglianza WBC richiede una decelerazione
  di almeno circa 1741 rad/s². Il log mostrava MPC/riferimenti finiti ma
  accelerazioni/forze WBC non finite. È compatibile con un problema numerico o
  di fattibilità del QP; non è una dimostrazione matematica di infeasibilità.
  Il mantenimento dell'ultima torque dopo un errore prolungato non è un
  controllo di recupero e può proseguire l'estensione. Non viene sostituito
  arbitrariamente con torque nulle, che non garantirebbero l'equilibrio.


#### Esito misurato: GUI per 179.6 secondi

Con la correzione della quota ruota destra e il probe leggero: **nessuna caduta
né NaN**, 99.25 Hz medi, inclinazione massima 0.34°, torque applicate nel 99.997%
dei confronti. Spostamento massimo 1.239 m, prevalentemente all'indietro.
Dopo il transitorio, velocità campionata circa -0.007 / -0.009 m/s; COM attorno
0.415 m rispetto al target 0.400 m. Il margine delle ginocchia resta circa
0.29–0.31 rad: non si osserva estensione progressiva verso il limite in questa
prova. Il robot deriva ma non peggiora con la distanza percorsa.

Questo esclude, **per questa prova**, una catena inevitabile “più distanza →
più errore numerico → caduta”. Non dimostra che il sistema regga a qualsiasi
carico o che l'assenza di sincronizzazione sia innocua. La prova precedente
falliva con circa 55–62 Hz; qui il feedback è vicino alla frequenza prevista.

Verifica separata: traslando stato, riferimenti e warm start di +10 m / -5 m,
la differenza massima delle torque è 1.39e-6 Nm. Non emerge una dipendenza
significativa dalla posizione assoluta accumulata. Il test è in
`tests/check_translation_invariance.py`.

Dati ripetibili:
- `test_results/drift_gui_180s.json`: riepilogo della prova;
- `test_results/drift_gui_180s.trace.json`: base, giunti e comandi nel tempo;
- `test_results/drift_gui_180s.urdf`: descrizione ROS usata;
- `test_results/drift_gui_180s.analysis.json`: finestre di 10 s con COM e margini;
- `test_results/drift_gui_180s_node.log`: frequenze e risultati del solver.

Analisi offline:

```bash
python3 src/mpx/tests/analyze_feedback_trace.py src/mpx/test_results/drift_gui_180s.json
```
Le velocità istantanee salvate a 20 Hz possono campionare sempre una fase simile
del ciclo a 100 Hz: la loro media non va scambiata automaticamente per un bias
fisico. La verifica successiva integra le velocità a ogni messaggio JointState.


#### Nuovo problema misurato: velocità e posizioni non coerenti

Nella prova GUI di 59.9 s il robot resta in piedi a circa 100 Hz, ma integrando
**tutti** i messaggi JointState (~500 Hz, ordine dei nomi invariato) il ginocchio
sinistro dà +30.519 rad contro -0.0131 rad di spostamento reale pubblicato.
Anche la ruota sinistra dà -16.387 rad integrati contro +4.238 rad misurati.
Non è solo aliasing della traccia a 20 Hz. I risultati sono salvati in
`test_results/drift_gui_velocity_60s.integrals.json`.

Il bridge legge direttamente `Position(0)` e `GetVelocity(0)` e registra i
puntatori sulle rispettive state interface: non è emerso uno scambio nel mapping.
Questo rende prematura qualsiasi conclusione “tutti i dati dinamici sono giusti”.
La cinematica a partire dalle posizioni può essere corretta mentre le velocità
lette producono uno stato dinamico incoerente. Verifica in corso: aumentare solo
le iterazioni ODE nell'istanza di test, mantenendo fisica a 500 Hz e GUI aperta,
per distinguere convergenza dei vincoli fisici e lettura/temporizzazione.


#### Causa verificata e correzione: convergenza ODE

A parità di GUI, passo 0.002 s e algoritmo MPX, portare soltanto il solver ODE
a 300 iterazioni cambia nettamente il risultato:

| Prova GUI ~60 s | Predefinito | 300 iterazioni |
|---|---:|---:|
| Massimo spostamento XY | 0.3255 m | 0.00261 m |
| Frequenza effort | 99.98 Hz | 99.68 Hz |
| Altezza finale base | 0.46530 m | 0.44480 m |
| Inclinazione finale | 0.2874° | 0.00366° |
| Errore integrale velocità ginocchio sinistro | 30.532 rad | 0.3452 rad |
| Errore integrale velocità ginocchio destro | 30.307 rad | 0.00494 rad |

Il miglioramento non deriva dall'aumento di priorità o da un riferimento di
posizione che maschera la deriva: il controller è lo stesso. È evidenza forte
che l'insufficiente convergenza dei vincoli ODE contribuisce sia allo stato
dinamico incoerente sia al bias del controllo. Il test non stabilisce che 300
iterazioni eliminino ogni errore numerico o garantiscano ogni scenario.

La modifica permanente è solo nel world `gazebo_bridge/worlds/empty_world.world`:
`physics/ode/solver/type=quick`, `iters=300`. Il passo e il rate restano invariati.
Le prove precedenti con meno iterazioni non dimostravano un errore di integrazione
nel nodo: il nodo rileggeva velocità problematiche dalla simulazione, e il WBC
le usava per COM, Jacobiani dinamici e vincoli di frenata.
La documentazione [Gazebo sui parametri fisici](https://classic.gazebosim.org/tutorials?tut=physics_params)
spiega il ruolo delle iterazioni nel solver QuickStep.

Artefatti del confronto: `test_results/drift_gui_velocity_60s.*` e
`test_results/drift_gui_iters300_60s.*`. Non sono state sostituite le velocità
con differenze finite nel controller e non sono stati modificati pesi MPC,
inerzie, posa iniziale o logica dei comandi per ottenere questo risultato.


#### Verifica finale della configurazione installata: GUI per 179.35 s

Il world con 300 iterazioni, ricaricato dopo il build, supera la prova lunga:

- **5.50 mm** di spostamento XY massimo (prima: 1.239 m in una prova di durata simile);
- inclinazione massima **0.256°**, finale **0.00367°**;
- base finale **0.44480 m**, nessun NaN/caduta;
- effort **96.81 Hz** medi, feedback giunti **499.64 Hz**;
- torque applicate nel **99.9989%** dei confronti; pausa/ripresa verificate.

Il controller non ha cambiato tuning per questa prova. La frequenza in alcune
finestre scende sotto 100 Hz: il miglioramento non richiede una simulazione
artificialmente bloccata in attesa di MPC. Restano errori numerici finiti e
feedback base/giunti non sincronizzato per stamp; non è una garanzia generale
di stabilità in ogni scenario. Non è stato necessario aggiungere un integratore
XY o sostituire le velocità con stime da differenze finite.

Artefatti finali: `test_results/drift_gui_iters300_180s.*`.
Grafico del confronto a pari durata:

![Confronto ODE, deriva e coerenza velocità](test_results/ode_velocity_comparison.png)

I test di frame/ordine giunti, riferimento asimmetrico ruote e pubblicazione
delle cinque stime passano. Build di `gazebo_bridge` e `mpx` completato.
Riavviare i processi già aperti dopo `source install/setup.bash` per caricare
world e codice aggiornati.

A regime, nella prova finale, il COM è circa **0.40009 m** (target 0.400 m),
contro circa 0.415 m prima della correzione. Il margine minimo delle ginocchia
si stabilizza intorno a **0.295 rad**. L'errore integrale residuo del ginocchio
sinistro è circa 0.360 rad in 180 s, vicino allo 0.345 rad della prova da 60 s:
è molto ridotto rispetto al caso precedente e non cresce allo stesso ritmo.
Questo non equivale a velocità numericamente perfette, ma distingue chiaramente
il vecchio errore persistente dal comportamento dopo la correzione.


## Confronto numerico XML standalone e xacro ROS — 26 settembre

Verificati i file correnti nel workspace, senza modificare il modello o il tuning.
`mpx_tita/mpx/examples/mjx_tita.py` carica `scene_flat.xml` (default), che include
`mpx/data/tita/tita.xml`. Il wrapper standalone carica separatamente lo stesso
`tita.xml` tramite `config.model_path`. Il nodo ROS passa invece il modello
costruito dal robot_description espanso: non carica quel file XML.

| Proprietà | XML standalone | Modello ROS da xacro |
|---|---|---|
| Massa totale | 27.68978 kg | 27.68978 kg |
| Masse e offset dei COM dei link | Coincidono | Coincidono |
| Cinematica, assi, centri ruota | Coincidono nei due assetti verificati | Differenze numeriche centri ruota <3e-16 m |
| Raggio / larghezza ruote | 0.0925 / 0.034 m | Uguali |
| Inerzie link leg_1, leg_2, leg_4 | Diagonali, orientamento inerziale identità | Tensore completo, prodotti d'inerzia non nulli |
| Inerzie base, leg_3 e IMU | Tensore conservato | Uguali |
| Armature degli otto giunti | 0.0005 kg m² | 0 |
| Limiti torque applicati | ±120 Nm | ±100 Nm |
| Giunti ruota | limited=false | Limiti ±62831.85 rad |
| Assetto articolare iniziale | [0, .5, -1, 0] per lato | Uguale con il launch attuale |
| Altezza iniziale base | Keyframe home: 0.440 m | Spawn launch: 0.443534 m |

I tensori sono confrontati nello stesso frame del link, ricostruendoli da
inerzie principali e orientamento: non si tratta soltanto di una diversa
rappresentazione XML del medesimo tensore. Differenza massima dei termini
inerziali circa 9.63e-5 kg m². A parità di stato, la matrice di massa differisce
fino a circa 5.94e-4 nei due assetti provati, mentre COM e contributo gravitazionale
coincidono entro precisione numerica. L'armature è inerzia aggiuntiva riflessa ai
giunti, non damping: può cambiare la risposta dinamica e il condizionamento.
Queste misure non quantificano da sole l'effetto sulla stabilità del controller.

Altre differenze di esperimento:

- Lo standalone legge posizione e velocità dal medesimo `MjData` e invoca
  `mj_step` dopo il controllo; Gazebo avanza indipendentemente da ROS/MPC.
- Contatti/frizione/solver sono quelli della scena MuJoCo contro quelli di ODE.
  I coefficienti solref/solimp di MuJoCo non sono parametri equivalenti del solver ODE.
- Il percorso ROS passa `com_height_target` al WBC; lo script standalone non lo
  passa e conserva il comportamento originale del wrapper. Quindi attualmente
  differiscono anche i riferimenti verticali, oltre al modello.
- La modifica ODE a 300 iterazioni ha ridotto la deriva mantenendo lo stesso
  modello ROS: il confronto precedente dimostra una causa indipendente dalle
  differenze XML/URDF. Non dimostra che queste ultime siano irrilevanti.

Conclusione: i modelli non sono identici dinamicamente. Non sarebbe corretto
attribuire tutto alla GPU, alla frequenza o all'XML senza isolare queste variabili.
Lo xacro rimane la fonte del modello ROS; il confronto non cambia automaticamente
inerzie o armature per copiare lo standalone.

Verifica ripetibile, dopo il source ROS/workspace:

```bash
python3 src/mpx/tests/compare_robot_models.py
```

Risultato dettagliato: `test_results/xml_vs_ros_model.json`. La verifica usa lo
xacro installato risolto da ament, cioè quello che il launch ROS trova nel workspace
sorgente del terminale; il percorso è riportato nel JSON. È un confronto dei
modelli compilati in MuJoCo, non una prova di equivalenza del motore fisico Gazebo.


### Leggibilità del passo MPC/WBC

`MPXNode.step_controller` segue la sequenza dello standalone `mjx_tita.py`:
`process_state` → `mpc.run` → `mpc.whole_body_run`, con variabili
`mpc_state`, `tita_state`, `x0`, `theta_prev`, `tau_wbc` e `qddot`.
`_step_controller_jit` è la versione compilata CPU della stessa funzione.
Il worker ROS conserva pianificazione temporale, controlli dei risultati non
finiti, diagnostica e commit solo dopo validazione. In caso di risultato invalido
restano lo stato MPC precedente e l'ultima torque valida applicata da ros2_control.
Il refactoring non cambia frequenze, algoritmo o riferimento verticale.


## Verifiche dopo il refactoring e tracking

Il refactoring mantiene la parità numerica delle torque (differenza zero nel
confronto con la sequenza MPC/WBC separata). La nuova prova GUI da 179.29 s,
`test_results/post_refactor_gui_180s.*`, termina senza NaN/cadute:
deriva massima 5.96 mm contro 5.50 mm prima, controllo medio 97.17 Hz contro
96.81 Hz. Una prova asincrona non è bit-per-bit ripetibile; non emerge una
regressione pratica della deriva.

Il confronto successivo esegue il vero `step_controller` di `mjx_tita.py`:
lo script accetta ora modelli e un osservatore opzionali, mantenendo invariata
la chiamata predefinita. `tests/check_mjx_model_sources.py` confronta robot XML
con robot da xacro sotto la stessa fisica MuJoCo, contatti, terreno e pose
iniziali. Il modello ROS di sola dinamica non è usato direttamente come scena:
vengono ripristinati pavimento, attuatori e contatti per poter simulare.
I confronti MuJoCo non vengono eseguiti contemporaneamente alle prove Gazebo,
per non falsare i tempi del controller.

Il test di tracking usa rampe massime 0.25 m/s² e 0.2 rad/s², con plateau fino
a vx=1.0 m/s e successivamente wz=0.4 rad/s. Misura twist nel frame di marcia
e variazioni di posa indipendenti. Non è un test di gradino istantaneo o una
garanzia di tutte le combinazioni dei due comandi.


### Standalone: confronto XML e robot da xacro

Il vero `mjx_tita.step_controller`, eseguito su CPU per 30 s simulati, mantiene
l'equilibrio sia con XML sia con il robot ricavato dallo xacro. La posizione
longitudinale finale è rispettivamente 2.6929 mm e 2.6724 mm: differenza di circa
0.0205 mm, senza NaN/cadute. Questo confronto non mostra un vantaggio sostanziale
dell'XML per l'equilibrio a comando nullo. Non implica identità dinamica né prova
di equivalenza per ogni manovra.

Il test mantiene identici terreno, opzioni del solver MuJoCo, parametri dei
contatti (incluse proprietà ereditate dai default MJCF), pose iniziali e tuning.
Nel caso ROS sostituisce robot e modello WBC con quelli derivati dall'URDF,
conservando i limiti effort 100 Nm e armature zero. Il ciclo `step_controller`
non è copiato: viene eseguito direttamente dal file dello standalone.

Artefatti: `test_results/mjx_xml_30s.*`, `test_results/mjx_ros_30s.*`.
La variante `ros_aligned` è un esperimento separato: sostituisce le inerzie
nel medesimo frame con quelle XML, imposta armature 0.0005 e limiti 120 Nm.
Non modifica lo xacro installato o i parametri della simulazione Gazebo.


### Esito dell'allineamento sperimentale XML

La variante da xacro allineata a inerzie/armature/limiti effort XML completa
30 s senza NaN o caduta: deriva massima 2.74785 mm, contro 2.74779 mm XML e
2.72741 mm xacro originale. L'altezza finale è circa 0.44467 m in tutti i casi.
La differenza non giustifica modificare permanentemente lo xacro per migliorare
l'equilibrio. La descrizione ROS conserva quindi i tensori completi e i limiti
fisici precedenti. Non è stato dichiarato un allineamento Gazebo dell'armature:
la variante sperimentale vive in MuJoCo e non aggiunge un'inerzia motore fittizia
al robot ROS.

Riproduzione dei tre confronti (uno per volta, a simulazione Gazebo spenta):

```bash
python3 src/mpx/tests/check_mjx_model_sources.py --source xml --seconds 30 --output /tmp/mjx_xml.json
python3 src/mpx/tests/check_mjx_model_sources.py --source ros --seconds 30 --output /tmp/mjx_ros.json
python3 src/mpx/tests/check_mjx_model_sources.py --source ros_aligned --seconds 30 --output /tmp/mjx_ros_aligned.json
```

Questa evidenza riguarda equilibrio a comando nullo, stessa posa iniziale e
stessi contatti; il tracking viene verificato separatamente con Gazebo e GUI.


### Tracking: correzione geometrica successiva al refactoring

Il primo test GUI completo supera vx=1.0 e wz=0.4 senza caduta/NaN.
Ai plateau lineari misura 0.20003, 0.50003 e 0.99767 m/s. Ai plateau angolari
misura 0.10618, 0.21146 e 0.41438 rad/s per comandi 0.1, 0.2 e 0.4.
La stima interna dello yaw rate rimaneva invece quasi esattamente sul comando.

Dalla traccia: la carreggiata nominale è 0.567 m, ma durante le rotazioni la
distanza orizzontale dei centri ruota è circa 0.534–0.548 m. La formula divideva
la differenza delle velocità dei centri per la distanza nominale, sottostimando
la rotazione reale. Usare la distanza dei contatti virtuali non è coerente con
queste velocità e lascia un errore aggiuntivo. Usando i centri per entrambi, sul
plateau 0.4 si ricostruiscono circa 0.41387 rad/s contro 0.41440 Gazebo.

`_process_state_impl` ora ricava heading e yaw rate dalla stessa linea tra i
centri ruota, usando la sua distanza orizzontale attuale. I contatti virtuali
restano usati per posizione del centro contatti; `config.d` resta nel modello
predittivo. È una correzione funzionale separata, successiva al refactoring
confermato invariato. Non è un cambio di guadagno o una taratura del comando.

Il test cinematico con distanza 0.45 m e yaw rate reale 0.4 prima restituiva
0.31746 rad/s; viene verificato anche con distanze 0.567 e 0.70 m.
La nuova versione è stata verificata nuovamente con gli stessi comandi GUI;
i risultati sono riportati sotto.
Artefatti della prima prova: `test_results/tracking_gui_vx1_wz04.*`.
In quell'arresto il processo MPX ha richiesto SIGTERM dopo il timeout SIGINT di
launch; i plateau erano già conclusi e la fisica messa in pausa. Questo evento
è distinto dall'esito del tracking e viene controllato nelle prove successive.


### Arresto in pausa dopo comandi esterni

In due prove di tracking concluse correttamente, MPX non terminava entro il
limite SIGINT di launch. In questa condizione il publisher esterno aveva fatto
rimuovere il timer del comando zero e la fisica era in pausa: senza callback,
`rclpy.spin` poteva rimanere nell'attesa DDS indefinita mentre il nodo usa il
gestore SIGINT Python. Il main usa ora un unico executor persistente con
`spin_once(timeout_sec=0.1)`: limita l'attesa senza eventi, senza imporre un
periodo di 100 ms alle callback o al solver. La prova `final_gui_shutdown.*` conferma
l'arresto pulito dopo comando esterno e pausa, senza escalation SIGTERM.


### Tracking GUI dopo la correzione della carreggiata

Prova `test_results/tracking_gui_measured_width.*`, solo CPU, con GUI aperta.
Le misure sono medie degli ultimi quattro secondi di ciascun plateau:

| Comando | Velocità misurata | RMSE sul plateau |
|---|---:|---:|
| vx = 0.2 m/s | 0.200027 m/s | 0.000027 m/s |
| vx = 0.5 m/s | 0.500027 m/s | 0.000027 m/s |
| vx = 1.0 m/s | 0.997644 m/s | 0.002452 m/s |
| wz = 0.1 rad/s | 0.100000 rad/s | < 0.000001 rad/s |
| wz = 0.2 rad/s | 0.200000 rad/s | < 0.000001 rad/s |
| wz = 0.4 rad/s | 0.400001 rad/s | < 0.000001 rad/s |

Il controllo indipendente da differenze di posa conferma circa 0.997971 m/s
al plateau lineare massimo e 0.400001 rad/s al plateau angolare massimo.
Nessun NaN/caduta; inclinazione massima dell'intera sequenza circa 0.2514°,
e torque applicate coerenti con quelle pubblicate nel 99.997% dei confronti.
Il test non dimostra precisione micrometrica di un robot reale: sono misure
interne alla simulazione, con terreno piano e rampe dei comandi.

Il bias angolare precedente (0.41438 rad/s a comando 0.4) è stato eliminato
correggendo l'ingresso di stato al controller, senza compensare il comando.
La deriva lineare residua a comando zero è piccola ma non nulla: dopo l'arresto
circa 0.000028 m/s in questa prova. Il controllo di velocità non garantisce
il ritorno alla posizione di partenza.

Riproduzione su una simulazione dedicata già avviata con GUI, in pausa e con
MPX pronto (il test invia comandi, avanza la fisica e infine la mette in pausa):

```bash
python3 src/mpx/tests/check_velocity_tracking.py --output /tmp/tracking.json
python3 src/mpx/tests/plot_velocity_tracking.py /tmp/tracking.json
```


### Verifica lunga finale con GUI

`test_results/final_gui_measured_width_180s.*` registra 179.732 s simulati:
deriva massima **5.615 mm**, inclinazione massima **0.246°**, inclinazione
finale **0.0036°**, altezza finale base **0.44480 m**. Nessun risultato invalido
nel log del controller e nessuna caduta. Frequenza media torque **99.85 Hz**,
JointState **499.93 Hz**, corrispondenza effort applicati **99.9989%**.
La pausa interrompe le pubblicazioni delle torque; alla ripresa riparte il
controllo sul tempo simulato. Circa l'80% dei campioni a 500 Hz conserva la stessa
torque, coerentemente con aggiornamento MPC/WBC a 100 Hz e mantenimento dell'ultimo
comando nel controller a 500 Hz.

La deriva rimane piccola ma non zero: questa prova copre tre minuti, terreno
piano e assetto iniziale documentato, non dimostra stabilità per durata illimitata.
Il probe di equilibrio ha exit code 0. La verifica aggiuntiva di shutdown nel
primo runner è fallita per un errore di sintassi nel solo comando di test; viene
ripetuta separatamente in `final_gui_shutdown.*`, con exit code 0 e processo
MPX terminato correttamente dopo aver rimosso il timer del comando predefinito.


### Regressione conclusiva dello standalone e degli ingressi

Dopo la correzione della carreggiata sono stati ripetuti i 30 s dello standalone:
`mjx_xml_final_30s.*` e `mjx_ros_final_30s.*`, entrambi senza fallimenti.
Derive massime rispettivamente 2.747793 mm e 2.727408 mm, sostanzialmente
identiche ai confronti precedenti. L'analisi offline della prova GUI finale
mostra COM z circa 0.40009 m e margine del ginocchio circa 0.29536 rad costanti
nei blocchi successivi al transitorio; non emerge un progressivo avvicinamento
ai limiti nell'intervallo provato.

I quattro test degli ingressi passano: frame delle velocità e ordine dei giunti,
riferimento verticale separato della ruota destra, publisher delle stime e yaw
rate con diverse distanze tra i centri ruota.

```bash
python3 -m unittest discover -s src/mpx/tests -p test_state_inputs.py -v
```

[Grafico del tracking con GUI](test_results/tracking_gui_measured_width.png).


### Altezza COM dalla tastiera

`mpx_keyboard` carica `config/keyboard.yaml`: Page Up/Down varia il riferimento
COM tra 0.30 e 0.40 m, passo 0.01 m, partenza 0.40 m. Il nodo mantiene i binding
ROS e usa incrementi lineare/angolare additivi di 0.1 dal YAML.
`/mpx/com_height` (`std_msgs/Float64`, transient local) aggiorna il quarto elemento
del comando MPC, già usato anche come `com_height_target` nel WBC. Il callback
`cmd_vel` conserva l'altezza scelta; Spazio ferma la marcia senza cambiarla.
La tastiera usa ora incrementi indipendenti: i/freccia su e k/freccia giù
modificano vx; l/freccia destra e j/freccia sinistra modificano wz.
Il controller rifiuta NaN e limita l'altezza a `config_tita.h_min/h_max`.
Riavviare MPX per caricare il nuovo subscriber.

Verifica: pseudo-terminale con subscriber ROS isolati, Page Up/Down, limiti,
incrementi e arresto; cinque test degli ingressi superati, inclusa la persistenza
dell'altezza tra messaggi Twist. Questa verifica riguarda l'interfaccia comandi;
non è una nuova prova dinamica Gazebo dell'intero intervallo di altezza.
