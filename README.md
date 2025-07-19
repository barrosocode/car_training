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

#

