# Cálculo de Métricas de Avaliação para Modelos de Classificação

## Descrição do Projeto

Este projeto tem como objetivo calcular as principais métricas usadas para avaliar modelos de classificação de dados, incluindo **acurácia**, **sensibilidade (recall)**, **especificidade**, **precisão** e **F-score**. A implementação dessas métricas será realizada utilizando suas respectivas fórmulas e métodos específicos.

A base para o cálculo dessas métricas será uma **matriz de confusão**, que pode ser escolhida arbitrariamente para facilitar o entendimento do funcionamento de cada métrica. O projeto oferece flexibilidade na seleção da matriz de confusão, permitindo explorar como diferentes matrizes influenciam as métricas calculadas.

## Métricas a Serem Calculadas

- **Acurácia**: A proporção de previsões corretas (verdadeiros positivos e verdadeiros negativos) sobre todas as previsões realizadas.
  
- **Sensibilidade (Recall)**: A taxa de verdadeiros positivos corretamente identificados pelo modelo.
  
- **Especificidade**: A taxa de verdadeiros negativos corretamente identificados pelo modelo.
  
- **Precisão**: A proporção de previsões verdadeiramente positivas em relação a todas as previsões positivas feitas pelo modelo.
  
- **F-score**: A média harmônica entre precisão e sensibilidade, equilibrando ambas as métricas.

## Matriz de Confusão

Uma matriz de confusão é usada para calcular essas métricas. A matriz contém os seguintes valores:

- **VP (Verdadeiros Positivos)**: Os casos onde o modelo previu corretamente uma classe positiva.
- **FN (Falsos Negativos)**: Os casos onde o modelo previu incorretamente uma classe negativa quando era, na verdade, positiva.
- **FP (Falsos Positivos)**: Os casos onde o modelo previu incorretamente uma classe positiva quando era, na verdade, negativa.
- **VN (Verdadeiros Negativos)**: Os casos onde o modelo previu corretamente uma classe negativa.

## Objetivo

O objetivo é entender como cada métrica funciona, implementando e calculando os valores com base na matriz de confusão escolhida.
