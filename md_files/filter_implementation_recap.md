# Filtro di stato della base per MPX: recap

Data: 2026-09-29

## In breve

- MPX ora può leggere lo stato della base (posa e velocità) da **due sorgenti**, scelte con un flag:
  - `ground_truth`: `/model_states` di Gazebo. Esiste **solo in simulazione**.
  - `filter`: `/filtered_state` del nuovo package `tita_state_estimator`. Funziona **in simulazione e sull'hardware**.
- In Gazebo, con `filter`, MPX completa la sequenza standard:
  `transform_up → rl_0 → mpc → vx=0.3 → curva → stop → rl_0`.
  - Non genera nessun comando invalido.
  - L'handoff a RL va a buon fine.
  - Il filtro sbaglia di circa **1 mm in altezza** e di **≤ 0.014 m/s in velocità** rispetto alla ground truth.
- Sull'hardware `hw.launch.py` avvia sempre il filtro e MPX con `mpx_base_state:=filter` (default)
  oppure `odometry` (vedi sotto).
  **Non è ancora stato provato sul robot**: vedi "Procedura sull'hardware".


## Come scegliere: ground truth, filtro o odometria

> Aggiornamento 2026-09-30: c'è una terza sorgente, `odometry`. Da
> `/tita4267305/chassis/odometry` (che non stima z) MPX prende x, y e la velocità
> orizzontale; dal filtro z, vz, orientamento e velocità angolare. Non ancora provata:
> in Gazebo quel topic non esiste (`checkpoints/test_tools/fake_chassis_odom.py` lo simula).

### In simulazione (Gazebo)

```bash
# MPX usa la ground truth di Gazebo (default, comportamento di prima)
ros2 launch rl_controller sim_gazebo.launch.py mpx_base_state:=ground_truth

# MPX usa il filtro
ros2 launch rl_controller sim_gazebo.launch.py mpx_base_state:=filter

# MPX usa odometria chassis + filtro
ros2 launch rl_controller sim_gazebo.launch.py mpx_base_state:=odometry
```

Il filtro (`state_estimator_node`) viene avviato **sempre** in Gazebo, anche con
`ground_truth`. In questo modo puoi confrontare `/filtered_state` con `/model_states`
mentre MPX usa la ground truth. Il flag cambia solo la sorgente che legge MPX.

### Parametro MPX (`src/mpx/config/mpx_node.yaml`)

```yaml
inputs:
  base_state_source: filter                    # "ground_truth", "filter" oppure "odometry"
  model_states_topic: /model_states            # usato con ground_truth
  filtered_state_topic: /filtered_state        # usato con filter e odometry
  odometry_topic: /tita4267305/chassis/odometry  # usato con odometry
```

Nei launch `config/tita/mpx.yaml` rende relativo `filtered_state`. I launch sovrascrivono `base_state_source` con il valore di `mpx_base_state`.
All'avvio MPX stampa `Base state source: ...`.

## Il package `tita_state_estimator`

Contiene il filtro di Kalman `StateFilter_no_bias.hpp`, estratto da
[TITA-dynamic-obstacle-avoidance](https://github.com/Emilianogith/TITA-dynamic-obstacle-avoidance)
(`tita_controller/include/`). È la versione usata sul loro TITA reale. Il repository
clonato non serve più: ha un `COLCON_IGNORE` e si può cancellare.

| File | Contenuto |
|---|---|
| `include/tita_state_estimator/StateFilter_no_bias.hpp` | Il filtro, identico all'originale (cambia solo l'`#include`) |
| `include/tita_state_estimator/wheel_geometry.hpp` | `get_rCP`, `compute_virtual_frame`, copiate da `utils.cpp` |
| `src/state_estimator_node.cpp` | Nodo ROS 2, solo la parte di stima di `controller_node.cpp` |

**Come funziona il filtro.** Lo stato è `[p_base, v_base, p_contatto_sx, p_contatto_dx]` nel frame mondo.
- **Predizione:** usa l'accelerometro per la base e la cinematica di rotolamento delle ruote per i contatti.
- **Correzione:** usa la cinematica diretta (base → punti di contatto) e assume i contatti a z = 0.
- **Orientamento e velocità angolare:** vengono direttamente dall'IMU.

**Nodo:**

| | |
|---|---|
| Input | `/imu_sensor_broadcaster/imu`, `/joint_states`, URDF da `/robot_description` |
| Output | `/filtered_state` (`nav_msgs/Odometry`): posa nel frame `odom`, twist (lineare e angolare) nel frame `base_link` |
| Frequenza | un passo per ogni messaggio `/joint_states` (500 Hz), con `dt` preso dai timestamp dei messaggi |
| Reset | `ros2 service call /state_estimator/reset std_srvs/srv/Trigger` |

**Parametri:** `max_input_age` (0.1 s), `gyro_offset` ([0, 0, 0]). I topic sono relativi e
vengono da `ros_utils/topic_names.hpp` (`imu_sensor_broadcaster/imu`, `joint_states`,
`robot_description`, `filtered_state`): il namespace del launch fa il resto.

**Differenze rispetto al nodo originale:**
- **`gyro_offset` a zero di default.** L'originale sommava `[-0.0017, 0.0036, 0.00023]`
  rad/s, una calibrazione del loro esemplare di robot. Col robot fermo quell'offset faceva
  derivare x di circa 0.25 m in 10 s. Sul vostro robot va calibrato: media del giroscopio a robot fermo, col segno cambiato.
- **Si avvia da solo** alla prima coppia IMU + giunti, invece di aspettare il servizio
  `start_filter`, e assume le ruote a terra. Dopo che il robot si è alzato conviene chiamare `reset`.
- **Si reinizializza** se i timestamp di IMU e giunti differiscono di più di `max_input_age`,
  o se il tempo torna indietro (reset della simulazione).
- **Non ci sono** controllore, log CSV, TF né percorsi hardcoded.

MPX riconverte il twist dal frame base al frame mondo (`on_filtered_state` in
`mpx_node.py`), così `update_mujoco_state` resta invariato.

## Risultati in Gazebo (headless)

Sequenza: `transform_up → rl_0 → reset filtro → mpc → vx=0.3 → curva (vx 0.2, wz 0.5) → stop → rl_0`.
Gli errori sono calcolati come filtro − `/model_states`; le velocità sono nel frame mondo.

### MPX con `mpx_base_state:=filter`

| Fase | z GT [m] | errore z medio | errore v RMS (x, y, z) [m/s] |
|---|---|---|---|
| RL fermo | 0.316 | −1.4 mm | 0.007, 0.001, 0.013 |
| RL vx=0.5 | 0.311 | −1.4 mm | 0.005, 0.001, 0.014 |
| MPC fermo | 0.441 | −1.1 mm | 0.000, 0.000, 0.014 |
| MPC vx=0.3 | 0.442 | −1.0 mm | 0.000, 0.000, 0.014 |
| MPC curva | 0.445 | −0.6 mm | 0.004, 0.003, 0.014 |
| RL dopo handoff | 0.319 | −1.0 mm | 0.002, 0.002, 0.014 |

Log di MPX:
- `COM at h_mpc=0.400 m: MPC commands unlocked`;
- 75–83 Hz, solve mediano 12 ms, **invalid=0**;
- `COM at h_rl=0.310 m: handing off to rl_0`.

### MPX con `mpx_base_state:=ground_truth` (riferimento)

- Stessa sequenza a vx=0.3: stabile, unlock e handoff regolari, invalid=0.

## Bug trovato e corretto durante i test

Nella prima versione il nodo girava con un timer a 500 Hz e usava `now()`. In Gazebo,
con `use_sim_time`, `now()` segue `/clock`, che `gazebo_ros` pubblica **a soli 10 Hz**:
- il filtro integrava con `dt = 0` per 99 tick su 100;
- poi faceva un salto di `dt = 0.1 s`;
- la velocità stimata restava ferma per 100 ms e poi saltava.

In anello aperto l'errore sembrava piccolo, ma **in anello chiuso MPX faceva cadere il
robot circa 0.4 s dopo la presa di controllo**, mentre con la ground truth restava in piedi.
Ora il filtro avanza a ogni `/joint_states` usando i timestamp dei messaggi, come fa già MPX.

## File modificati

| File | Modifica |
|---|---|
| `tita_state_estimator/` (nuovo) | package del filtro |
| `mpx/scripts/mpx_node.py` | `inputs.base_state_source`; sottoscrizione `Odometry` + `on_filtered_state`; tutti i topic letti dal config (niente più `cmd_vel`, `com_height`, `/mpx/estimate` scritti nel codice) |
| `mpx/config/mpx_node.yaml` | `base_state_source: ground_truth`, `filtered_state_topic: /filtered_state`, `cmd_vel_topic`, `com_height_topic`, `estimate_topic_prefix` |
| `mpx/package.xml` | `depend nav_msgs` |
| `mpx/scripts/mpx_node.py` (attesa dati) | dopo `mpc`, se mancano stato base o giunti: `MPX waiting for data on: <topic>` ogni 2 s |
| `interaction/keyboard_controller/` | se il subscriber `*_rl_controller` su `/command/cmd_key` sparisce, la modalità torna a `idle` (controllo ogni 0.5 s) |
| `controller/rl_controller/launch/sim_gazebo.launch.py` | argomento `mpx_base_state`; avvio di `state_estimator_node` (solo TITA); `inputs.cmd_vel_topic` al posto del remapping `/mpx/cmd_vel` |
| `controller/rl_controller/launch/hw.launch.py` | argomento `enable_mpx`; con MPX avvia anche `state_estimator_node`; MPX con `base_state_source: filter`; `inputs.cmd_vel_topic` al posto del remapping |
| `controller/rl_controller/package.xml` | `exec_depend tita_state_estimator` |
| `TITA-dynamic-obstacle-avoidance/COLCON_IGNORE` | esclude il repo clonato dalla build |

Rispetto al checkpoint `checkpoints/mpx_baseline_2026-09-28`, `sha256sum -c` segnala anche
`rl_controller/CMakeLists.txt`, `hardware_bridge_node.cpp` e `keyboard_controller.cpp`.
Non li ho toccati in questa sessione (modifiche del network watchdog e precedenti).

## Procedura sull'hardware

Prerequisiti:
- workspace compilato sul robot, incluso `tita_state_estimator`;
- Pinocchio installato (qui è in `/opt/openrobots`).

```bash
colcon build --packages-select tita_state_estimator mpx rl_controller
ros2 launch rl_controller hw.launch.py            # enable_mpx true per tita, namespace:=$ROBOT_NS
```

1. **IMU.**
   `ros2 topic hz /imu_sensor_broadcaster/imu` deve dare circa 500 Hz.
   Da fermo in piano: giroscopio circa 0, `linear_acceleration.z` circa +9.8.
2. **Filtro senza MPX** (non premere `4`: MPX resta in standby e non manda coppie).
   - Alza il robot con `7` e passa in RL con `0`.
   - Poi `ros2 service call /state_estimator/reset std_srvs/srv/Trigger`.
   - Controlla `ros2 topic echo /filtered_state`:
     - `position.z` da fermo circa 0.31 m (altezza base in RL, come in Gazebo);
     - `twist.linear` circa 0 da fermo;
     - `twist.linear.x` coerente guidando in avanti in RL.
   - Se x e y derivano da fermo, calibra `gyro_offset`: media del giroscopio a robot
     fermo, col segno cambiato, passata come parametro del nodo nel launch.
3. **Prima prova MPC.** Robot sospeso o e-stop pronto; conviene ridurre `torque_limit` in
   `rl_controller/config/tita/controllers.yaml`.
   - Premi `4` e aspetta la compilazione (in Gazebo circa 30–55 s, sul robot da misurare).
   - Controlla nel log di MPX:
     - `Base state source: filter`;
     - `COM at h_mpc=0.400 m: MPC commands unlocked`;
     - `Control: X Hz ... solve ms median=...`: serve circa 10 ms o meno, e invalid=0.
   - Poi prova piccole velocità e l'handoff con `0`.
4. **Se qualcosa va storto:** premi `0` (RL) o `6` (idle).
   Attenzione: se MPX si blocca mentre è in MPC, `FSMState_MPC` continua ad applicare
   l'ultima coppia ricevuta finché non cambi modalità.

## Limiti noti del filtro

- **Assume sempre le ruote a terra** (`in_contact = true`). Se il robot salta o viene
  sollevato, altezza e velocità stimate sono sbagliate finché non torna a terra.
- **x e y sono odometria pura:** derivano nel tempo e ripartono da 0 a ogni `reset`.
  MPX lavora in velocità, quindi non è un problema.
- La gravità è fissa nel filtro: `g = 9.81` (nell'originale 9.744, il valore misurato dal
  loro accelerometro). Anche il world di Gazebo ora ha `<gravity>0 0 -9.81</gravity>`.
- **Come sorgente di MPX in moto** il filtro ha fatto cadere il robot 2 volte su 2 (0 su 3
  con la ground truth): la sua velocità diverge quando le ruote rimbalzano.
- **Il filtro dipende dall'orientamento dell'IMU.** Un errore di assetto si riflette
  direttamente su altezza e velocità stimate.
