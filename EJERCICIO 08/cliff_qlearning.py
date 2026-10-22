import random
import csv

# dimensiones y posiciones del tablero del libro (4x12)
ROWS = 4
COLS = 12
START = (3, 0)
GOAL = (3, 11)
CLIFF = {(3, c) for c in range(1, 11)}

# orden fijo de acciones para cuando haya empates en la tabla Q
ACTIONS = ['N', 'E', 'S', 'W']
ACTION_DELTAS = {
    'N': (-1, 0),
    'E': (0, 1),
    'S': (1, 0),
    'W': (0, -1)
}

def step(state, action):
    # aqui simulamos el movimiento del agente en el mapa
    r, c = state
    dr, dc = ACTION_DELTAS[action]
    
    # si choca con una pared se queda en su lugar
    nr = max(0, min(ROWS - 1, r + dr))
    nc = max(0, min(COLS - 1, c + dc))
    next_state = (nr, nc)

    # si cae al acantilado te regresa a start con -100 pero NO termina el juego
    if next_state in CLIFF:
        return START, -100, False
    
    # si llega a la meta da -1 y si termina el episodio
    if next_state == GOAL:
        return GOAL, -1, True
    
    # cualquier otro paso normal da -1 y sigue la partida
    return next_state, -1, False


# funciones del agente q-learning
def empty_q():
    # inicializa toda la tabla q en ceros
    q = {}
    for r in range(ROWS):
        for c in range(COLS):
            q[(r, c)] = {a: 0.0 for a in ACTIONS}
    return q

def argmax_action(q, state):
    # busca la accion con el valor mas alto, respetando el orden fijo si hay empate
    best_action = ACTIONS[0]
    best_val = q[state][best_action]
    for a in ACTIONS[1:]:
        if q[state][a] > best_val:
            best_val = q[state][a]
            best_action = a
    return best_action

def epsilon_greedy(q, state, eps):
    # decide si explora al azar o si usa lo que ya aprendio
    if random.random() < eps:
        return random.choice(ACTIONS)
    return argmax_action(q, state)

def td_target_q(q, next_state, reward, gamma, is_terminal):
    # calcula la formula de q-learning para actualizar la tabla
    if is_terminal:
        return float(reward)
    max_next_q = max(q[next_state].values())
    return reward + gamma * max_next_q

def apply_update(q, state, action, target, alpha):
    # aplica el ajuste a la casilla q(s,a)
    q[state][action] += alpha * (target - q[state][action])


# validaciones y entrenamiento
def verify_hand_cases():
    # checa los tres casos que pedia el profe para estar seguros
    assert step((3, 0), 'N') == ((2, 0), -1, False)
    assert step((3, 0), 'E') == ((3, 0), -100, False)
    assert step((2, 11), 'S') == ((3, 11), -1, True)
    print("casos de prueba manuales pasaron bien\n")

def train(episodes=500, alpha=0.5, gamma=1.0, eps=0.1, max_steps=200, seed=42):
    random.seed(seed)
    q = empty_q()
    episode_rewards = []

    for ep in range(episodes):
        state = START
        total_reward = 0
        for _ in range(max_steps):
            action = epsilon_greedy(q, state, eps)
            next_state, reward, is_terminal = step(state, action)
            
            target = td_target_q(q, next_state, reward, gamma, is_terminal)
            apply_update(q, state, action, target, alpha)
            
            state = next_state
            total_reward += reward
            if is_terminal:
                break
        episode_rewards.append(total_reward)

    return q, episode_rewards

def evaluate_greedy(q, max_steps=200):
    # corre el agente sin azar (eps = 0) para ver el camino optimo final
    state = START
    path = [state]
    actions_taken = []
    total_reward = 0

    for _ in range(max_steps):
        action = argmax_action(q, state)
        next_state, reward, is_terminal = step(state, action)
        
        actions_taken.append(action)
        path.append(next_state)
        total_reward += reward
        state = next_state
        
        if is_terminal:
            break

    return path, actions_taken, total_reward

def print_policy_grid(q):
    # imprime el mapa con las flechitas de direccion
    arrows = {'N': '↑', 'E': '→', 'S': '↓', 'W': '←'}
    print("cuadricula de politica greedy:")
    print("    " + " ".join(f"{c:2d}" for c in range(COLS)))
    for r in range(ROWS):
        row_str = f"{r:2d}  "
        for c in range(COLS):
            pos = (r, c)
            if pos == START:
                cell = " S"
            elif pos == GOAL:
                cell = " G"
            elif pos in CLIFF:
                cell = " C"
            else:
                cell = " " + arrows[argmax_action(q, pos)]
            row_str += cell + " "
        print(row_str)
    print()

def moving_average(values, window=10):
    # saca la media movil para suavizar los datos de la grafica
    ma = []
    for i in range(len(values)):
        start_idx = max(0, i - window + 1)
        chunk = values[start_idx : i + 1]
        ma.append(sum(chunk) / len(chunk))
    return ma


# ejecucion principal del programa
if __name__ == "__main__":
    SEED = 42
    ALPHA = 0.5
    GAMMA = 1.0
    EPS = 0.1
    EPISODES = 500
    MAX_STEPS = 200

    verify_hand_cases()

    print(f"entrenando con semilla {SEED}...")
    q_table, rewards = train(EPISODES, ALPHA, GAMMA, EPS, MAX_STEPS, SEED)
    ma_rewards = moving_average(rewards, window=10)

    path, actions, greedy_return = evaluate_greedy(q_table)
    
    print_policy_grid(q_table)
    print("resultado de la evaluacion greedy:")
    print(f"acciones: {' -> '.join(actions)}")
    print(f"pasos totales: {len(actions)}")
    print(f"retorno greedy: {greedy_return}")

    # guardamos en csv
    with open("recompensas_entrenamiento.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episodio", "Recompensa", "Media_Movil_10"])
        for i, (r, ma) in enumerate(zip(rewards, ma_rewards), start=1):
            writer.writerow([i, r, round(ma, 2)])
    print("\ndatos guardados en csv.")

    # intentamos generar la grafica si esta matplotlib
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, EPISODES + 1), rewards, alpha=0.3, color='steelblue', label='recompensa cruda')
        plt.plot(range(1, EPISODES + 1), ma_rewards, color='darkblue', linewidth=2, label='media movil (10)')
        plt.axhline(-13, color='crimson', linestyle='--', label='optimo (-13)')
        plt.ylim(-150, 0)
        plt.xlabel('episodios')
        plt.ylabel('recompensa')
        plt.title('q-learning cliff walking')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("curva_qlearning.png", dpi=150)
        print("grafica generada como 'curva_qlearning.png'.")
    except ImportError:
        print("matplotlib no disponible, solo se guardo el csv.")