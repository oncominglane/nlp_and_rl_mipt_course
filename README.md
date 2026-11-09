# NLP & RL MIPT Course

Материалы курса NLP&RL (по обработке естественного языка и обучению с подкреплением)
от кафедры машинного обучения и цифровой гуманитаристики МФТИ, 8-й семестр.

В репозитории собраны лекции, практические задания и экзаменационные материалы. Краткие описания лекций ниже составлены по содержанию файла [`exam/NLP&RL.tex`](exam/NLP&RL.tex).

## Структура репозитория

- [`lectures/`](lectures/) - презентации лекций курса.
- [`seminars/`](seminars/) - ноутбуки семинарских занятий.
- [`tasks/`](tasks/) - практические задания и шаблоны решений.
- [`exam/`](exam/) - экзаменационный конспект, вопросы, ответы и иллюстрации.

## Лекции

1. [**Лекция 1 - Word Embeddings**](lectures/01_word_embeddings_dssm_lecture.pdf) - введение в задачи NLP и способы представления текста.  
   Основные понятия: токенизация, нормализация, стемминг, лемматизация, Bag-of-Words, Bag-of-Ngrams, коллокации, PMI, pPMI, TF-IDF, one-hot encoding, distributional semantics, матричная факторизация, word embeddings, Word2Vec, Skip-gram, CBOW, negative sampling, GloVe.

2. [**Лекция 2 - CNN for Texts and Embeddings for Different Languages**](lectures/02_CNN_for_texts_and_more_embeddings.pdf) - эмбеддинги для разных языков и сверточные сети для текстов.  
   Основные понятия: cross-lingual embeddings, линейное и ортогональное отображение между языками, unsupervised mapping, cosine similarity, embedding текста, RNN recap, переход от RNN к CNN, n-граммы, свертка по тексту, one-layer CNN, архитектура Kim 2014, static и non-static embeddings, narrow и wide convolution.

3. [**Лекция 3 - Machine Translation and Attention Mechanism**](lectures/03_machine_translation_attention.pdf) - машинный перевод от статистических подходов к нейросетевым моделям с attention.  
   Основные понятия: Statistical Machine Translation, word alignment, Phrase-Based Machine Translation, Neural Machine Translation, Seq2Seq, encoder-decoder, greedy decoding, exhaustive search, beam search, length normalization, BLEU, human evaluation, bottleneck Seq2Seq, attention, dot-product/multiplicative/additive attention.

4. [**Лекция 4 - Self-Attention and Transformer**](lectures/04_transformer.pdf) - устройство Transformer и роль self-attention.  
   Основные понятия: attention score, encoder-decoder Transformer, encoder block, self-attention, Query/Key/Value, матричная форма attention, Multi-Head Attention, positional encoding, sinusoidal positional encoding, Layer Normalization, residual connections, Feed-Forward Network, masked self-attention, encoder-decoder attention, сравнение RNN/CNN/Transformer.

5. [**Лекция 5 - Transfer Learning, BERT, GPT**](lectures/05_bert_and_transfer_learning_intro.pdf) - transfer learning в NLP и предобученные трансформерные модели.  
   Основные понятия: self-attention recap, Multi-Head Attention, positional encoding, Layer Normalization, transfer learning, OpenAI Transformer и GPT-подход, fine-tuning, input transformations, ELMo, bidirectional language models, BERT, BERT Base/Large, BERT inputs, Masked Language Modeling, Next Sentence Prediction, BERT fine-tuning, BERT tokenization.

6. [**Лекция 6 - Contrastive Learning**](lectures/06_contrastive_learning.pdf) - self-supervised и contrastive learning для обучения представлений.  
   Основные понятия: self-supervised learning, pretext task, generative vs self-supervised learning, оценка self-supervised признаков, rotation prediction, jigsaw puzzles, inpainting, image coloring, split-brain autoencoder, contrastive representation learning, positive и negative samples, InfoNCE, cosine similarity, SimCLR, projection head, large batch size, MoCo, momentum update, Contrastive Predictive Coding, mutual information.

7. [**Лекция 7 - Introduction to Reinforcement Learning**](lectures/07_Intro_to_RL.pdf) - базовая постановка обучения с подкреплением и первые алгоритмы.  
   Основные понятия: agent-environment loop, policy, reward, Multi-Armed Bandits, MDP, Markov property, total reward, value function, Q-function, Bellman relations, Cross-Entropy Method, tabular CEM, approximate CEM, continuous action space, Q-learning, approximate Q-learning, Basic Deep Q-Learning, reward formulation.

8. [**Лекция 8 - Bellman Equations**](lectures/08_Bellman_equations.pdf) - функции ценности, уравнения Беллмана и динамическое программирование в RL.  
   Основные понятия: MDP, reward hypothesis, return, reward discounting, discount factor, reward design, reward scaling, reward shaping, expected objective, backup tree, state-value function, action-value function, Bellman expectation equations, optimal policy, Bellman optimality equations, policy evaluation, policy improvement, generalized policy iteration, policy iteration, value iteration.

9. [**Лекция 9 - Model-Free Learning in Reinforcement Learning**](lectures/09_Model_free_learning.pdf) - обучение без модели среды и методы на основе траекторий.  
   Основные понятия: model-based vs model-free learning, Monte Carlo estimation, Temporal Difference learning, Q-learning, Monte Carlo vs TD, policy из Q-function, exploration-exploitation tradeoff, approximate Q-learning, MSE formulation, архитектуры approximate Q-learning, Basic Deep Q-Learning.

10. [**Лекция 10 - Policy Gradient and REINFORCE**](lectures/10_policy_gradient.pdf) - policy-based методы и градиентная оптимизация политики.  
    Основные понятия: ограничения value-based методов, deterministic и stochastic policies, Cross-Entropy Method как policy-based метод, policy gradient, objective для one-step process, log-derivative trick, REINFORCE, discounted returns, on-policy learning, variance reduction, baselines, advantage function, Actor-Critic, Advantage Actor-Critic, continuous action spaces, entropy regularization, A3C, IMPALA.

11. [**Лекция 11 - Reinforcement Learning in Language Modeling and SCST**](lectures/11_RL_in_Language_modeling.pdf) - применение RL к sequence generation и языковым моделям.  
    Основные понятия: RL-формализм для генерации последовательностей, encoder-decoder architectures, supervised seq2seq learning, attentive translation, sequence generation, distribution shift, неоднозначность правильного ответа, conversation systems, seq2seq как POMDP, policy gradient для sequence generation, cold start problem, training vs inference mismatch, Self-Critical Sequence Training, SCST baseline, image captioning with SCST, discrete GANs, RL fine-tuning.

## Семинарские ноутбуки

- [`practice1_01_dealing_with_word_embeddings__completed.ipynb`](seminars/practice1_01_dealing_with_word_embeddings__completed.ipynb) - обучение, использование и визуализация word embeddings.
- [`practice02_cnn_for_texts__completed.ipynb`](seminars/practice02_cnn_for_texts__completed.ipynb) - предсказание зарплаты по тексту вакансии с помощью CNN.
- [`practice1_03_seq2seq_nmt_and_tensorboard__completed.ipynb`](seminars/practice1_03_seq2seq_nmt_and_tensorboard__completed.ipynb) - модель Seq2Seq для машинного перевода и визуализация обучения в TensorBoard.
- [`practice1_04_extra_attention_basics_and_tensorboard.ipynb`](seminars/practice1_04_extra_attention_basics_and_tensorboard.ipynb) - основы реализации механизма attention.
- [`practice1_04_seq2seq_nmt__with_attention.ipynb`](seminars/practice1_04_seq2seq_nmt__with_attention.ipynb) - машинный перевод с помощью Seq2Seq и attention.
- [`week05_bilstm_for_pos_tagging__completed.ipynb`](seminars/week05_bilstm_for_pos_tagging__completed.ipynb) - определение частей речи с помощью BiLSTM.
- [`bert_for_text_classification.ipynb`](seminars/bert_for_text_classification.ipynb) - классификация текстов с использованием признаков DistilBERT.
- [`bert_finetuning.ipynb`](seminars/bert_finetuning.ipynb) - дообучение BERT для классификации предложений на наборе CoLA.
- [`practice_question_answering_and_tts.ipynb`](seminars/practice_question_answering_and_tts.ipynb) - ответы на вопросы с помощью BERT и озвучивание ответов.
- [`practice_pagerank.ipynb`](seminars/practice_pagerank.ipynb) - пример реализации алгоритма PageRank.
- [`pytorch_and_dataloaders.ipynb`](seminars/pytorch_and_dataloaders.ipynb) - основы PyTorch, загрузка данных и обучение модели.

## Практические задания

- `tasks/task1` - задания по word embeddings и language modeling.
- `tasks/task2` - задания по attention, классической классификации текстов, BERT и переводу.
- `tasks/task3` - задания по reinforcement learning: Cross-Entropy Method, Q-learning и policy gradient.

