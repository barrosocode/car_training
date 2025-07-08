# import gymnasium as gym
# from gymnasium.core import Wrapper
# import numpy as np
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import VecFrameStack

# # --- Custom Wrapper para Terminar o Episódio na Grama ---
# # ESTE WRAPPER DEVE SER IDÊNTICO AO USADO NO ARQUIVO DE TREINAMENTO.
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
#             obs, reward, terminated, info = step_result
#             truncated = False
#         else:
#             raise ValueError(f"O método step() do ambiente base retornou um número inesperado de valores: {len(step_result)}")

#         # --- Lógica de Término Imediato na Grama (Idêntica ao treinamento) ---
#         green_channel = obs[:, :, 1]
#         red_channel = obs[:, :, 0]
#         blue_channel = obs[:, :, 2]

#         grass_pixels = (green_channel > 120) & \
#                        (green_channel > red_channel * 1.2) & \
#                        (green_channel > blue_channel * 1.2) & \
#                        (red_channel < 100) & (blue_channel < 100)

#         num_grass_pixels = np.sum(grass_pixels)

#         TERMINATE_GRASS_THRESHOLD = 50
#         OFF_TRACK_PENALTY = -500

#         if num_grass_pixels > TERMINATE_GRASS_THRESHOLD:
#             reward = OFF_TRACK_PENALTY
#             terminated = True
#             info['off_track_by_grass'] = True

#         # --- Aumentar Recompensa por Estar na Pista (Idêntica ao treinamento) ---
#         ON_TRACK_BONUS = 0.05
#         if num_grass_pixels < TERMINATE_GRASS_THRESHOLD / 2:
#              reward += ON_TRACK_BONUS
#         else:
#             reward -= (num_grass_pixels / 9216.0) * 0.5


#         self.last_reward_raw = reward
#         return obs, float(reward), terminated, truncated, info

#     def reset(self, **kwargs):
#         return self.env.reset(**kwargs)

# # --- Configuração do Ambiente de Teste (Idêntica ao treinamento, exceto render_mode e n_envs) ---
# def make_env_with_wrappers_for_test():
#     env = gym.make("CarRacing-v3", continuous=True, render_mode="human") # human para visualização
#     env = CustomCarRacingWrapper(env)
#     return env

# eval_env_base = make_vec_env(make_env_with_wrappers_for_test, n_envs=1) # n_envs=1 para um único ambiente
# eval_env_final = VecFrameStack(eval_env_base, n_stack=4)


# # --- Carregar o Modelo Treinado (sem alterações) ---
# try:
#     model_path = "./car_racing_ppo_final_model.zip"
#     model = PPO.load(model_path)
#     print(f"Modelo carregado com sucesso de: {model_path}")
# except FileNotFoundError:
#     print(f"Erro: Modelo não encontrado em {model_path}. Verifique se o caminho e o nome do arquivo estão corretos.")
#     print("Certifique-se de que o treinamento foi executado e salvou o modelo.")
#     exit()
# except Exception as e:
#     print(f"Erro ao carregar o modelo: {e}")
#     exit()

# # --- Testar o Modelo ---
# print("Iniciando teste do modelo treinado...")

# obs_batch = eval_env_final.reset() # Captura o array de observações

# try:
#     for step_count in range(5000): # Roda por 5000 passos
#         action, _states = model.predict(obs_batch, deterministic=True)

#         # Desempacota assumindo a API de 4 valores para o VecEnv.step()
#         obs_batch, rewards, done_array, infos = eval_env_final.step(action)

#         eval_env_final.render("human")

#         # Verifica se o episódio terminou (para o único ambiente no batch)
#         if done_array[0]:
#             print(f"Episódio de avaliação terminado no passo {step_count+1}. Recompensa final: {rewards[0]:.2f}")
#             obs_batch = eval_env_final.reset() # Reseta o ambiente

# except KeyboardInterrupt:
#     print("Teste interrompido pelo usuário.")
# finally:
#     eval_env_final.close()
#     print("Ambiente de avaliação fechado.")

# ===================================================================

import gymnasium as gym
from gymnasium.core import Wrapper
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

# --- Custom Wrapper para Terminar o Episódio na Grama ---
# Este wrapper DEVE ser idêntico ao usado no treinamento para consistência.
class CustomCarRacingWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.last_reward_raw = 0

    def step(self, action):
        step_result = self.env.step(action)

        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        elif len(step_result) == 4:
            obs, reward, terminated, info = step_result # 'terminated' aqui é o antigo 'done'
            truncated = False # Assume truncated como False se a API for a antiga
        else:
            raise ValueError(f"O método step() do ambiente base retornou um número inesperado de valores: {len(step_result)}")

        # --- Lógica de Término Imediato na Grama ---
        green_pixels = (obs[:, :, 1] > 180) & (obs[:, :, 0] < 100) & (obs[:, :, 2] < 100)
        num_green_pixels = np.sum(green_pixels)

        GRASS_PIXEL_THRESHOLD = 500
        OFF_TRACK_PENALTY = -1000

        if num_green_pixels > GRASS_PIXEL_THRESHOLD:
            reward = OFF_TRACK_PENALTY
            terminated = True
            info['off_track_by_grass'] = True

        self.last_reward_raw = reward
        return obs, float(reward), terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

# --- Configuração do Ambiente de Teste ---
def make_env_with_wrappers_for_test():
    env = gym.make("CarRacing-v3", continuous=True, render_mode="human")
    env = CustomCarRacingWrapper(env)
    return env

eval_env_base = make_vec_env(make_env_with_wrappers_for_test, n_envs=1)
eval_env_final = VecFrameStack(eval_env_base, n_stack=4)


# --- Carregar o Modelo Treinado ---
try:
    # model_path = "./car_racing_ppo_final_model_colab_pessoal.zip" # Verifique se este é o nome correto do seu modelo
    # model_path = "./car_racing_ppo_final_model_colab_academico.zip"
    model_path = "./car_racing_model_400000_steps.zip"
    # model_path = "./car_racing_model_800000_steps.zip"

    model = PPO.load(model_path)
    print(f"Modelo carregado com sucesso de: {model_path}")
except FileNotFoundError:
    print(f"Erro: Modelo não encontrado em {model_path}. Verifique se o caminho e o nome do arquivo estão corretos.")
    print("Certifique-se de que o treinamento foi executado e salvou o modelo.")
    exit()
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    exit()

# --- Testar o Modelo ---
print("Iniciando teste do modelo treinado...")

obs_batch = eval_env_final.reset() # Captura o array de observações

try:
    for step in range(5000): # Roda por 5000 passos para ver o desempenho
        action, _states = model.predict(obs_batch, deterministic=True)

        # Desempacota assumindo a API de 4 valores para o VecEnv.step()
        obs_batch, rewards, done_array, infos = eval_env_final.step(action)

        eval_env_final.render("human")

        # Verifica se o episódio terminou (para o único ambiente no batch)
        if done_array[0]: # done_array[0] é True se o episódio terminou por qualquer motivo
            print(f"Episódio de avaliação terminado no passo {step+1}. Recompensa final: {rewards[0]:.2f}")
            # Reseta o ambiente e pega a nova observação.
            obs_batch = eval_env_final.reset()

except KeyboardInterrupt:
    print("Teste interrompido pelo usuário.")
finally:
    eval_env_final.close()
    print("Teste concluído. Ambiente de avaliação fechado.")
