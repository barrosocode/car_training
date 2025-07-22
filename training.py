import gymnasium as gym
from gymnasium.core import Wrapper
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecFrameStack

#teste
class CustomCarRacingWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.last_reward_raw = 0

    def step(self, action):
        # Captura o resultado do step do ambiente base
        step_result = self.env.step(action)

        # Adapta-se à API do Gymnasium (4 ou 5 valores)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        elif len(step_result) == 4:
            obs, reward, terminated, info = step_result # 'terminated' aqui é o antigo 'done'
            truncated = False # Assume truncated como False se a API for a antiga
        else:
            raise ValueError(f"O método step() do ambiente base retornou um número inesperado de valores: {len(step_result)}")

        # --- Lógica de Término Imediato na Grama ---
        # Heurística para identificar pixels de grama:
        green_pixels = (obs[:, :, 1] > 180) & (obs[:, :, 0] < 100) & (obs[:, :, 2] < 100)
        num_green_pixels = np.sum(green_pixels)

        # Se uma quantidade significativa de grama for detectada, termine o episódio
        # O limiar (ex: 500 pixels) e a penalidade (-1000) podem ser ajustados.
        GRASS_PIXEL_THRESHOLD = 500 # Quantidade de pixels de grama para considerar "fora da pista"
        OFF_TRACK_PENALTY = -1000 # Penalidade severa por sair da pista

        if num_green_pixels > GRASS_PIXEL_THRESHOLD:
            reward = OFF_TRACK_PENALTY # Aplica uma penalidade alta
            terminated = True         # Termina o episódio
            # Opcional: Adicionar uma informação para depuração
            info['off_track_by_grass'] = True

        self.last_reward_raw = reward # Para depuração

        # O 'float(reward)' é importante para garantir o tipo correto,
        # pois o Stable Baselines3 espera recompensas como float.
        return obs, float(reward), terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

# --- Configurações de Treinamento ---
N_ENVS = 4
TOTAL_TIMESTEPS = 2_400_000 # Lembre-se de aumentar isso para milhões para treinamento real
SAVE_FREQ = 100000

# --- 1. Criação e Empilhamento dos Ambientes ---
def make_env_with_wrappers():
    env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
    env = CustomCarRacingWrapper(env)
    return env

vec_env = make_vec_env(make_env_with_wrappers, n_envs=N_ENVS, seed=0)
vec_env = VecFrameStack(vec_env, n_stack=4)

# --- 2. Criação do Modelo PPO ---
model = PPO("CnnPolicy", vec_env, verbose=1,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            max_grad_norm=0.5,
            tensorboard_log="./car_racing_ppo_tensorboard/")

# --- 3. Callbacks ---
checkpoint_callback = CheckpointCallback(
    save_freq=SAVE_FREQ,
    save_path="./car_racing_ppo_models/",
    name_prefix="car_racing_model"
)

# --- 4. Treinamento ---
print("Iniciando treinamento...")
model.learn(total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback)
print("Treinamento concluído!")

# --- 5. Salvar o Modelo Final ---
model.save("car_racing_ppo_final_model")
print("Modelo final salvo como car_racing_ppo_final_model.zip")

# ======================================================================

# import gymnasium as gym
# from gymnasium.core import Wrapper
# import numpy as np
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.callbacks import CheckpointCallback
# from stable_baselines3.common.vec_env import VecFrameStack

# # --- Custom Wrapper com Lógica de Recompensa e Término Melhoradas ---
# class CustomCarRacingWrapper(Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.env = env
#         self.last_reward_raw = 0

#     def step(self, action):
#         step_result = self.env.step(action)

#         if len(step_result) == 5:
#             obs, reward, terminated, truncated, info = step_result
#         elif len(step_result) == 4:
#             obs, reward, terminated, info = step_result # 'terminated' aqui é o antigo 'done'
#             truncated = False
#         else:
#             raise ValueError(f"O método step() do ambiente base retornou um número inesperado de valores: {len(step_result)}")

#         # --- Heurística Aprimorada para Grama ---
#         # A grama no CarRacing é predominantemente verde.
#         # Vamos ser um pouco mais flexíveis nos tons para capturar mais grama.
#         # Os pixels são RGB (0-255).
#         # Verdes: canal G alto, R e B baixos.
#         # Podemos aumentar o limiar do verde e diminuir dos outros canais.

#         # Detectar pixels de grama
#         # Heurística: G > R*1.5 e G > B*1.5 e G > (algum limiar mínimo)
#         # E que R e B sejam relativamente baixos.
#         green_channel = obs[:, :, 1]
#         red_channel = obs[:, :, 0]
#         blue_channel = obs[:, :, 2]

#         # Considera pixels "verdes" se o canal verde for significativamente maior que vermelho/azul
#         # e o verde em si for acima de um certo limiar (para evitar tons escuros de grama).
#         grass_pixels = (green_channel > 120) & \
#                        (green_channel > red_channel * 1.2) & \
#                        (green_channel > blue_channel * 1.2) & \
#                        (red_channel < 100) & (blue_channel < 100)

#         num_grass_pixels = np.sum(grass_pixels)

#         # --- Lógica de Término Imediato na Grama ---
#         # Reduzir o threshold para terminar o jogo mais cedo.
#         # Se for muito baixo, qualquer poeira verde pode terminar o jogo.
#         # Ajuste esse valor com base na observação visual do seu carro entrando na grama.
#         TERMINATE_GRASS_THRESHOLD = 50 # Número de pixels de grama para "game over"
#         OFF_TRACK_PENALTY = -500 # Penalidade severa. Originalmente -100 por morte, -500 é bem punitivo.

#         if num_grass_pixels > TERMINATE_GRASS_THRESHOLD:
#             reward = OFF_TRACK_PENALTY # Aplica uma penalidade alta
#             terminated = True         # Termina o episódio imediatamente
#             info['off_track_by_grass'] = True
#             # print(f"DEBUG: Terminated due to grass! Pixels: {num_grass_pixels}") # Para depuração

#         # --- Aumentar Recompensa por Estar na Pista (Adição) ---
#         # Se o carro NÃO estiver na grama (ou seja, está na pista ou quase), dê um bônus.
#         # Isso complementa a recompensa de tiles do ambiente base e incentiva a permanência.
#         # O inverso de num_grass_pixels. Se a grama é muito pouca, adicione recompensa.
#         # Pode ser um pequeno bônus por frame.
#         ON_TRACK_BONUS = 0.05 # Recompensa por frame por estar na pista
#         if num_grass_pixels < TERMINATE_GRASS_THRESHOLD / 2: # Se tem pouquíssima grama (dentro da pista)
#              reward += ON_TRACK_BONUS # Recompensa por estar limpo na pista
#         else: # Se está com pouca grama, mas não o suficiente para game over, penaliza um pouco mais.
#             reward -= (num_grass_pixels / 9216.0) * 0.5 # Penalidade suave baseada na proporção de grama

#         self.last_reward_raw = reward # Para depuração

#         return obs, float(reward), terminated, truncated, info

#     def reset(self, **kwargs):
#         return self.env.reset(**kwargs)

# # --- Configurações de Treinamento (sem alterações) ---
# N_ENVS = 4
# TOTAL_TIMESTEPS = 90_000 # Lembre-se: ainda é um valor de teste. Aumente para milhões para um bom treino.
# SAVE_FREQ = 45000

# # --- Criação e Empilhamento dos Ambientes (sem alterações) ---
# def make_env_with_wrappers():
#     env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
#     env = CustomCarRacingWrapper(env) # Agora com a nova lógica
#     return env

# vec_env = make_vec_env(make_env_with_wrappers, n_envs=N_ENVS, seed=0)
# vec_env = VecFrameStack(vec_env, n_stack=4)

# # --- Criação do Modelo PPO (sem alterações) ---
# model = PPO("CnnPolicy", vec_env, verbose=1,
#             learning_rate=0.0003,
#             n_steps=2048,
#             batch_size=64,
#             n_epochs=10,
#             gamma=0.99,
#             gae_lambda=0.95,
#             clip_range=0.2,
#             ent_coef=0.01,
#             max_grad_norm=0.5,
#             tensorboard_log="./car_racing_ppo_tensorboard/")

# # --- Callbacks (sem alterações) ---
# checkpoint_callback = CheckpointCallback(
#     save_freq=SAVE_FREQ,
#     save_path="./car_racing_ppo_models/",
#     name_prefix="car_racing_model"
# )

# # --- Treinamento (sem alterações) ---
# print("Iniciando treinamento...")
# model.learn(total_timesteps=TOTAL_TIMESTEPS,
#             callback=checkpoint_callback)
# print("Treinamento concluído!")

# # --- Salvar o Modelo Final (sem alterações) ---
# model.save("car_racing_ppo_final_model")
# print("Modelo final salvo como car_racing_ppo_final_model.zip")

# ============================================================

# import gymnasium as gym
# from gymnasium.core import Wrapper
# import numpy as np
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.callbacks import CheckpointCallback
# from stable_baselines3.common.vec_env import VecFrameStack

# # --- Custom Wrapper para Terminar o Episódio na Grama ---
# class CustomCarRacingWrapper(Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.env = env
#         self.last_reward_raw = 0

#     def step(self, action):
#         # Captura o resultado do step do ambiente base
#         step_result = self.env.step(action)

#         # Adapta-se à API do Gymnasium (4 ou 5 valores)
#         if len(step_result) == 5:
#             obs, reward, terminated, truncated, info = step_result
#         elif len(step_result) == 4:
#             obs, reward, terminated, info = step_result # 'terminated' aqui é o antigo 'done'
#             truncated = False # Assume truncated como False se a API for a antiga
#         else:
#             raise ValueError(f"O método step() do ambiente base retornou um número inesperado de valores: {len(step_result)}")

#         # --- Lógica de Término Imediato na Grama ---
#         # Heurística para identificar pixels de grama:
#         green_pixels = (obs[:, :, 1] > 180) & (obs[:, :, 0] < 100) & (obs[:, :, 2] < 100)
#         num_green_pixels = np.sum(green_pixels)

#         # Se uma quantidade significativa de grama for detectada, termine o episódio
#         # O limiar (ex: 500 pixels) e a penalidade (-1000) podem ser ajustados.
#         GRASS_PIXEL_THRESHOLD = 500 # Quantidade de pixels de grama para considerar "fora da pista"
#         OFF_TRACK_PENALTY = -1000 # Penalidade severa por sair da pista

#         if num_green_pixels > GRASS_PIXEL_THRESHOLD:
#             reward = OFF_TRACK_PENALTY # Aplica uma penalidade alta
#             terminated = True         # Termina o episódio
#             # Opcional: Adicionar uma informação para depuração
#             info['off_track_by_grass'] = True

#         self.last_reward_raw = reward # Para depuração

#         # O 'float(reward)' é importante para garantir o tipo correto,
#         # pois o Stable Baselines3 espera recompensas como float.
#         return obs, float(reward), terminated, truncated, info

#     def reset(self, **kwargs):
#         return self.env.reset(**kwargs)

# # --- Configurações de Treinamento (Permanecem as mesmas) ---
# N_ENVS = 4
# TOTAL_TIMESTEPS = 90_000 # Lembre-se de aumentar isso para milhões para treinamento real
# SAVE_FREQ = 45000

# # --- 1. Criação e Empilhamento dos Ambientes (Permanecem as mesmas) ---
# def make_env_with_wrappers():
#     env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
#     env = CustomCarRacingWrapper(env)
#     return env

# vec_env = make_vec_env(make_env_with_wrappers, n_envs=N_ENVS, seed=0)
# vec_env = VecFrameStack(vec_env, n_stack=4)

# # --- 2. Criação do Modelo PPO (Permanecem as mesmas) ---
# model = PPO("CnnPolicy", vec_env, verbose=1,
#             learning_rate=0.0003,
#             n_steps=2048,
#             batch_size=64,
#             n_epochs=10,
#             gamma=0.99,
#             gae_lambda=0.95,
#             clip_range=0.2,
#             ent_coef=0.01,
#             max_grad_norm=0.5,
#             tensorboard_log="./car_racing_ppo_tensorboard/")

# # --- 3. Callbacks (Permanecem as mesmas) ---
# checkpoint_callback = CheckpointCallback(
#     save_freq=SAVE_FREQ,
#     save_path="./car_racing_ppo_models/",
#     name_prefix="car_racing_model"
# )

# # --- 4. Treinamento (Permanecem as mesmas) ---
# print("Iniciando treinamento...")
# model.learn(total_timesteps=TOTAL_TIMESTEPS,
#             callback=checkpoint_callback)
# print("Treinamento concluído!")

# # --- 5. Salvar o Modelo Final (Permanecem as mesmas) ---
# model.save("car_racing_ppo_final_model")
# print("Modelo final salvo como car_racing_ppo_final_model.zip")

# # Note: A seção de teste deve ser separada em um arquivo 'test_model.py'
# # conforme a última instrução.
