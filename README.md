# Aprendizado de Caminho
*Deep Reinforcement Learning em Veículos Autônomos

[![Status](https://github.com/barrosocode/car_training/actions/workflows/blank.yml/badge.svg)](https://github.com/barrosocode/car_training/actions/workflows/blank.yml) [![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/barrosocode/car_training/blob/main/LICENSE) ![Linguagem](https://img.shields.io/github/languages/top/barrosocode/car_training)

## Resumo

Este projeto consiste no desenvolvimento de um agente de direção autônoma utilizando Aprendizagem Profunda por Reforço(Deep Reinforcement Learning). O agente foi treinado para percorrer a pista do ambiente *CarRacing* do começo ao fim, aprendendo as políticas de controle ótimas através de tentativa e erro.

**Principais tecnologias e métodos utilizados:**

* **Ambiente:** Construído e gerenciado com a biblioteca **`Gymnasium (gym)`**.
* **Algoritmo:** Proximal Policy Optimization (**PPO**) da **`Stable Baselines3`**.
* **Otimização de Treinamento:** Vetorização de ambientes com `make_vec_env` para paralelismo e **`VecFrameStack`** para fornecer memória visual ao agente.
* **Pré-processamento:** Wrappers customizados (`gym.Wrapper`) e manipulação de arrays com **`NumPy`**.
* **Persistência do Modelo:** Salvamento de checkpoints durante o treino com **`CheckpointCallback`**.

## Introdução
**Trabalho da Unidade III - Disciplina: Visão Computacional 2025.1**

A direção autônoma representa um dos desafios mais significativos e transformadores no campo da inteligência artificial e da engenharia. O desenvolvimento de veículos capazes de navegar em ambientes complexos de forma segura e eficiente tem o potencial de revolucionar os sistemas de transporte, reduzir acidentes e otimizar o fluxo de tráfego. Abordagens tradicionais para essa tarefa frequentemente dependem de um conjunto massivo de regras pré-programadas e sensores caros, tornando-as frágeis e de difícil generalização para cenários inesperados.

Como uma alternativa poderosa, a **Aprendizagem Profunda por Reforço (DRL)** surge como uma metodologia promissora, inspirada na forma como humanos aprendem: através da interação e da experiência. Em vez de seguir instruções explícitas, um agente de DRL aprende a tomar as melhores ações por meio de um sistema de recompensas e penalidades, otimizando sua política de decisão ao longo do tempo.

Este projeto aplica os conceitos de DRL para treinar um agente a dirigir um carro no ambiente de simulação **`CarRacing-v2`** da biblioteca `Gymnasium`. Este ambiente é particularmente desafiador por exigir que o agente aprenda a controlar o veículo (ações contínuas de volante, aceleração e freio) baseando-se unicamente em dados visuais brutos (pixels), simulando o que uma câmera a bordo de um carro real capturaria.

Os objetivos específicos deste trabalho são:
1.  Configurar e customizar o ambiente `CarRacing-v2` para o treinamento de um agente de DRL.
2.  Implementar e treinar um agente utilizando o algoritmo **Proximal Policy Optimization (PPO)**, reconhecido por sua robustez e eficiência em ambientes de controle contínuo.
3.  Desenvolver um modelo final capaz de completar a pista de forma autônoma, demonstrando um comportamento de direção coerente.
4.  Documentar o processo, a arquitetura e os resultados como um guia prático para a aplicação de `Stable Baselines3` em problemas de controle baseados em visão.

## Metodologia

Esta seção detalha a arquitetura do sistema de treinamento e a organização estrutural do código-fonte, que foram projetadas para garantir um fluxo de trabalho claro e reprodutível.

### Arquitetura do Sistema

A arquitetura do projeto é centrada no ciclo de interação padrão de um agente de Aprendizagem por Reforço, onde o Agente e o Ambiente trocam informações continuamente. Esse processo é orquestrado pela biblioteca `Stable Baselines3`, que abstrai grande parte da complexidade do loop de treinamento.

O fluxo de dados e processos pode ser visualizado no diagrama abaixo:

````mermaid
graph TD;
    A["Ambiente Gymnasium (CarRacing-v2)"] -- Observação (Frame) --> B["Pré-processamento (Wrappers)"];
    B -- Estado Processado --> C["Agente PPO (Rede Neural)"];
    C -- "Ações (Volante, Acel., Freio)" --> A;
    C -- "Dados de Treino" --> D["Loop de Treinamento (Stable Baselines3)"];
    D -- "Atualização de Pesos" --> C;
    style A fill:#11111,stroke:#333,stroke-width:2px;
    style C fill:#11111,stroke:#333,stroke-width:2px;
````


# Resultados

# Instalação
### 1. Pré-requisitos

Antes de instalar as dependências Python, certifique-se de ter os seguintes softwares instalados:

* **Python 3.8 ou superior**: [Download Python](https://www.python.org/downloads/)
* **Git**: [Download Git](https://git-scm.com/downloads/)

#### **Pré-requisitos Específicos do Sistema Operacional:**

<details>
<summary><strong>Para Linux (baseado em Debian/Ubuntu)</strong></summary>

Você precisará das ferramentas de compilação essenciais e da biblioteca SWIG, que são dependências para o ambiente `CarRacing`.

```bash
sudo apt-get update
sudo apt-get install -y build-essential swig
```
</details>

<details>
<summary><strong>Para Windows</strong></summary>

A instalação no Windows requer algumas ferramentas de compilação C++ para a biblioteca `Box2D`.

1.  **Microsoft C++ Build Tools**:
    * Faça o download do [Visual Studio Installer](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
    * Execute o instalador e, na aba "Cargas de Trabalho", selecione a opção **"Desenvolvimento para desktop com C++"**.
    * Prossiga com a instalação.

2.  **SWIG**:
    * Faça o download do **SWIG for Windows** (procure por `swigwin`) no [site oficial](http://swig.org/download.html).
    * Descompacte o arquivo (ex: em `C:\swigwin`).
    * Adicione a pasta descompactada ao **PATH** do seu sistema para que o `pip` possa encontrá-la.
        * Pesquise por "Editar as variáveis de ambiente do sistema" no Windows.
        * Clique em "Variáveis de Ambiente...".
        * Na seção "Variáveis do sistema", selecione a variável `Path` e clique em "Editar".
        * Clique em "Novo" e adicione o caminho para a pasta do SWIG (ex: `C:\swigwin`).
</details>
<br>

---

### 2. Instalação do Projeto

Siga os passos abaixo no seu terminal ou PowerShell.

**a. Clone o repositório:**
```bash
git clone [https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git](https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git)
cd SEU_REPOSITORIO
```

**b. Crie e ative um ambiente virtual (Recomendado):**
```bash
# Cria o ambiente virtual
python -m venv venv

# Ativa o ambiente
# No Windows (PowerShell):
venv\Scripts\Activate.ps1
# No Linux ou Windows (Git Bash):
source venv/bin/activate
```

**c. Instale as bibliotecas Python:**

O arquivo `requirements.txt` original contém pacotes específicos de GPU para Linux. **Não o utilize diretamente**. Siga a opção correspondente ao seu hardware.

<details>
<summary><strong>Opção A: Instalação com GPU NVIDIA (Recomendado para Treino)</strong></summary>

Esta opção utiliza a aceleração da sua placa de vídeo NVIDIA para um treinamento muito mais rápido.

1.  **Instale o PyTorch com suporte a CUDA:**
    Visite o [site oficial do PyTorch](https://pytorch.org/get-started/locally/) para obter o comando de instalação exato para sua versão do CUDA. Para CUDA 12.1, o comando geralmente é:
    ```bash
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    ```

2.  **Instale as outras dependências:**
    ```bash
    pip install stable_baselines3[extra] gymnasium[box2d] tensorboard
    ```
    * `stable_baselines3[extra]` instala a biblioteca com suas dependências comuns.
    * `gymnasium[box2d]` garante a instalação correta do ambiente `CarRacing`.

</details>

<details>
<summary><strong>Opção B: Instalação apenas com CPU</strong></summary>

Use esta opção se você não tem uma placa de vídeo NVIDIA ou não deseja configurar o CUDA. O treinamento será significativamente mais lento.

1.  **Instale a versão CPU do PyTorch:**
    ```bash
    pip install torch torchvision torchaudio
    ```

2.  **Instale as outras dependências:**
    ```bash
    pip install stable_baselines3[extra] gymnasium[box2d] tensorboard
    ```
</details>
<br>

---

### 3. Execução

**a. Para treinar um novo modelo:**

Execute o script de treinamento. O progresso será exibido no terminal, e os modelos e logs serão salvos nas pastas `car_racing_ppo_models/` e `car_racing_ppo_tensorboard/`[cite: 2].
```bash
python training.py
```
* O script `training.py` está configurado para executar 2.400.000 passos de tempo e salvar um checkpoint a cada 100.000 passos[cite: 2].
* O modelo usa a política "CnnPolicy" e vários hiperparâmetros como `learning_rate=0.0003`, `gamma=0.99` e `batch_size=64`[cite: 2].

**b. Para avaliar um modelo já treinado:**

Execute o script de teste. Uma janela do Pygame aparecerá mostrando o carro sendo controlado pelo agente.
```bash
python test.py
```
* **Importante**: Antes de executar, abra o arquivo `test.py` e verifique se a variável `model_path` aponta para o modelo (`.zip`) que você deseja testar.
* O ambiente de teste é configurado para renderizar em modo `human` para visualização.

---

### Contribuidores

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/Ag0ds"><img src="https://github.com/Ag0ds.png?size=100" alt="Foto de Gabriel Arthur"/><br/><sub><b>Gabriel Arthur</b></sub></a><br/><sub>Desenvolvedor</sub>
    </td>
    <td align="center">
        <a href="https://github.com/barrosocode"><img src="https://github.com/barrosocode.png?size=100" alt="Foto de SGabriel Barroso"/><br/><sub><b>Gabriel Barroso</b></sub></a><br/><sub>Desenvolvedor</sub>
    </td>
    <td align="center">
        <a href="https://github.com/heltonmaia"><img src="https://github.com/heltonmaia.png?size=100" alt="Foto de Helton Maia"/><br/><sub><b>Helton Maia</b></sub></a><br/><sub>Orientador</sub>
    </td>
  </tr>
</table>

---
