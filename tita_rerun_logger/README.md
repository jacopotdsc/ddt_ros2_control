# tita_rerun_logger

Nodo ROS 2 (Humble) che registra in [Rerun](https://rerun.io) (`.rrd`) i dati utili al debug di
TITA con RL, MPX e il filtro di stato, in Gazebo e sul robot. Di default lavora **offline**
(solo file, nessun viewer), così toglie meno CPU a MPX.

Portato da `~/Desktop/save_packages/tita_rerun_logger` e adattato a questo workspace: topic
relativi (namespace del robot), filtro `tita_state_estimator`, FSM di `rl_controller`, coppie
di MPX su `mpx/effort`.

## Le sorgenti dei dati

Ogni grandezza confrontabile è registrata come `<grandezza>/<asse>/<sorgente>`: un grafico
per asse, una linea (con colore fisso) per sorgente.

| Sorgente | Colore | Da dove | Dove esiste |
|---|---|---|---|
| `gt` | verde | ground truth di Gazebo: `/link_states` + masse dell'URDF | solo simulazione |
| `filter` | blu | `filtered_state` (`tita_state_estimator`) + cinematica diretta dei `joint_states` | sim e robot |
| `imu` | viola | robot: `imu_sensor_broadcaster/imu` | sim e robot |
| `odom` | magenta | robot: `/tita4267305/chassis/odometry` (x, y, velocità planare, yaw rate) | solo robot |
| `mpx` | arancione | quello che MPX crede: `mpx/estimate/*`, lo stato passato a MPC/WBC | quando MPX controlla |
| `cmd` | grigio | comandi: `command/cmd_twist`, `command/cmd_pose` | sempre |

Lo stato `mpx` viene dalla sorgente scelta con `mpx_base_state` (`ground_truth`, `filter`,
`odometry`) più i giunti: con `ground_truth` coincide con `gt`, con `filter` mostra cosa
riceve davvero il controllore.

Errori rispetto alla ground truth: `errors/<sorgente>/<grandezza>/<asse>` e `.../norm`
(gli angoli sono riportati in [−π, π]).

## Cosa viene registrato

| Entity path | Contenuto | Sorgenti |
|---|---|---|
| `base/position/{x,y,z}` | posizione base (il filtro è nel frame `odom`: x, y ripartono da 0 a ogni reset) | gt, filter, odom |
| `base/rpy/{roll,pitch,yaw}` | orientamento base | gt, filter, imu, mpx, cmd |
| `base/vel_heading/{x,y,z}` | velocità base, x/y nel frame di heading, z nel mondo | gt, filter, odom, cmd |
| `base/ang_vel_body/{x,y,z}` | velocità angolare nel frame base | gt, filter, imu, odom, cmd |
| `com/{x,y,z}`, `com_vel/{x,y,z}` | CoM e sua velocità (`com/z/cmd` = altezza comandata) | gt, filter, mpx, cmd |
| `wheels/{left,right}/{x,y,z}` | centri ruota | gt, filter, mpx |
| `balance/com_ahead_of_wheels` | CoM − punto medio delle ruote, nel frame di heading: la grandezza che MPC regola | gt, filter, mpx |
| `balance/com_lateral`, `com_height_over_wheels`, `base_height_over_wheels` | idem | gt, filter, (mpx) |
| `joints/position/<j>`, `joints/velocity/<j>` | giunti (`joint_states`) | robot |
| `joints/effort/<j>/{measured,mpx_cmd}` | coppia misurata/applicata e coppia comandata da MPX | robot, mpx |
| `joints/effort_ratio/<j>` | \|coppia misurata\| / `torque_limit` di `controllers.yaml` (1 = clamp) | |
| `joints/effort_clamp/<j>` | comando MPX − coppia misurata | |
| `fsm/state`, `fsm/requested` | stato FSM (`fsm_state`) e modo richiesto (`command/cmd_key`), come codice numerico | |
| `mpx/keys_locked` | blocco tasti di MPX (0/1) | |
| `mpx/rate_hz/{estimate,effort}` | frequenza dei solve e delle coppie pubblicate (tempo simulato) | |
| `mpx/latency_ms` | tempo simulato tra il campione dei giunti usato e la pubblicazione (vedi limiti) | |
| `command/twist/*`, `command/pose/*` | comandi da tastiera/remoto | |
| `robot/imu/{ang_vel,lin_acc}` | IMU grezza | |
| `extra/<topic>/...` | ogni campo numerico di `extra_topics` (`/performance_metrics`, sul robot `motors_status`, batterie) | |
| `events` | log testuale: cambi di stato FSM, tasti, handoff, blocco tasti | |
| `world/...` | robot 3D (mesh) dalla GT, CoM e ruote di gt/filtro/MPX, scie | |

Codici FSM: `idle=0, transform_down=1, transform_up=2, joint_pd=3, rl_0/rl_flat=4, …, mpc=8`
(anche nel log `events`).

La timeline è `sim_time` (stamp dei messaggi). In Gazebo `/clock` esce a 10 Hz, quindi i
messaggi senza stamp (`/link_states`, `mpx/effort`) prendono il tempo del `joint_states`
pubblicato nello stesso istante (timestamp DDS). Sul robot la registrazione parte da t = 0.

## Uso

```bash
cd ~/ddt_ros2_ws && colcon build --symlink-install --packages-select tita_rerun_logger
source install/setup.bash

# terminale 1: sim (o hw.launch.py sul robot)
ros2 launch rl_controller sim_gazebo.launch.py
# terminale 2: tastiera
ros2 run keyboard_controller keyboard_controller_node
# terminale 3: logger (Ctrl+C per chiudere il file)
ros2 launch tita_rerun_logger rerun_logger.launch.py
# sul robot
ros2 launch tita_rerun_logger rerun_logger.launch.py use_sim_time:=false namespace:=$ROBOT_NS
```

Argomenti: `namespace`, `use_sim_time`, `nice` (default 10), `open_viewer`, `viewer_url`,
`save_rrd`, `rrd_dir`, `config`. Ogni avvio crea
`rerun_log/tita_rerun:AAMMGG_HHMMSS.rrd` (ora locale di inizio).

Rivedere i log (ROS non serve, basta `pip install --user rerun-sdk`):

```bash
ros2 run tita_rerun_logger view_logs            # ultima registrazione
ros2 run tita_rerun_logger view_logs --list     # elenco
ros2 run tita_rerun_logger view_logs -n 3       # ultime 3 insieme
python3 src/tita_rerun_logger/scripts/view_logs.py <file.rrd>   # senza colcon
```

Tab del viewer: Overview, Base pose, Base velocity, CoM & wheels, Balance, Errors vs GT,
Joints, Torques, MPX & commands, Robot / extra; a sinistra la vista 3D e gli eventi.

## Configurazione (`config/rerun_logger.yaml`)

- `topics.*`: nomi relativi (seguono il namespace), come `rl_controller/config/tita/mpx.yaml`.
  Assoluti solo `/link_states` e i topic dell'SDK del robot (`/tita4267305/...`, come in
  `mpx_node.yaml`).
- Ordine dei giunti di `mpx/effort` e `torque_limit`: letti da `controllers.yaml` di
  `rl_controller` (`controllers_yaml`, `controller_name`).
- `rates_hz.*`: frequenze di registrazione (tempo dei messaggi). Stima e coppie di MPX sono
  registrate tutte.
- `poll_period_s` (0.02): i topic veloci (500 Hz) non hanno una callback per messaggio: un
  timer legge a lotti le code (50 messaggi). Con una callback per messaggio il logger usava più
  di un core.
- `flush_period_s` (2.0): ogni quanto i valori accumulati vengono scritti in Rerun, in blocchi
  (un `rr.log` per valore costava troppa CPU e spazio su disco).

## Costo e limiti

- **CPU.** Misurato in Gazebo con GUI e `rtf:=0.8`: logger ~50% di un core (80% in MPC). Con
  la stessa prova breve `7 → 0 → 4 → 0 → 8`, MPX ha girato a 92–95 Hz senza logger, 73–76 Hz
  col logger a priorità normale e 80–81 Hz con `nice 10` (default). Con la CPU già al limite
  (vedi CLAUDE.md) il logger può quindi togliere a MPX il margine sopra ~75 Hz: per le prove
  delicate di presa di controllo meglio non usarlo, oppure lanciarlo su un altro PC.
- File: circa 25–30 MB al minuto di prova (più ~6 MB di mesh, una volta sola).
- `mpx/latency_ms` è approssimata: `joint_state_broadcaster` pubblica da un thread
  realtime, che con la CPU carica pubblica in ritardo. Mediana corretta (~12 ms, come il solve),
  ma circa il 5% dei campioni esce negativo o troppo grande.
- I punti di contatto sono stimati come centro ruota − raggio (terreno piano).
- Il filtro nel 3D è nel frame `odom`: coincide col mondo solo finché non deriva.
- Massa totale dall'URDF: 25.69 kg (stampata all'avvio).
