# NLP & RL MIPT Course

Материалы курса NLP&RL (по обработке естественного языка и обучению с подкреплением)
от кафедры машинного обучения и цифровой гуманитаристики МФТИ, 8-й семестр.

В репозитории собраны лекции, практические задания и экзаменационные материалы. Краткие описания лекций ниже составлены по содержанию файла [`exam/NLP&RL.tex`](exam/NLP&RL.tex).

## Структура репозитория

- [`lectures/`](lectures/) - презентации лекций курса.
- [`tasks/`](tasks/) - практические задания и шаблоны решений.
- [`exam/`](exam/) - экзаменационный конспект, вопросы, ответы и иллюстрации.

## Лекции

1. **Лекция 1 - Word Embeddings** - введение в задачи NLP и способы представления текста.  
   Основные понятия: токенизация, нормализация, стемминг, лемматизация, Bag-of-Words, Bag-of-Ngrams, коллокации, PMI, pPMI, TF-IDF, one-hot encoding, distributional semantics, матричная факторизация, word embeddings, Word2Vec, Skip-gram, CBOW, negative sampling, GloVe.

2. **Лекция 2 - CNN for Texts and Embeddings for Different Languages** - эмбеддинги для разных языков и сверточные сети для текстов.  
   Основные понятия: cross-lingual embeddings, линейное и ортогональное отображение между языками, unsupervised mapping, cosine similarity, embedding текста, RNN recap, переход от RNN к CNN, n-граммы, свертка по тексту, one-layer CNN, архитектура Kim 2014, static и non-static embeddings, narrow и wide convolution.

3. **Лекция 3 - Machine Translation and Attention Mechanism** - машинный перевод от статистических подходов к нейросетевым моделям с attention.  
   Основные понятия: Statistical Machine Translation, word alignment, Phrase-Based Machine Translation, Neural Machine Translation, Seq2Seq, encoder-decoder, greedy decoding, exhaustive search, beam search, length normalization, BLEU, human evaluation, bottleneck Seq2Seq, attention, dot-product/multiplicative/additive attention.

4. **Лекция 4 - Self-Attention and Transformer** - устройство Transformer и роль self-attention.  
   Основные понятия: attention score, encoder-decoder Transformer, encoder block, self-attention, Query/Key/Value, матричная форма attention, Multi-Head Attention, positional encoding, sinusoidal positional encoding, Layer Normalization, residual connections, Feed-Forward Network, masked self-attention, encoder-decoder attention, сравнение RNN/CNN/Transformer.

5. **Лекция 5 - Transfer Learning, BERT, GPT** - transfer learning в NLP и предобученные трансформерные модели.  
   Основные понятия: self-attention recap, Multi-Head Attention, positional encoding, Layer Normalization, transfer learning, OpenAI Transformer и GPT-подход, fine-tuning, input transformations, ELMo, bidirectional language models, BERT, BERT Base/Large, BERT inputs, Masked Language Modeling, Next Sentence Prediction, BERT fine-tuning, BERT tokenization.

6. **Лекция 6 - Contrastive Learning** - self-supervised и contrastive learning для обучения представлений.  
   Основные понятия: self-supervised learning, pretext task, generative vs self-supervised learning, оценка self-supervised признаков, rotation prediction, jigsaw puzzles, inpainting, image coloring, split-brain autoencoder, contrastive representation learning, positive и negative samples, InfoNCE, cosine similarity, SimCLR, projection head, large batch size, MoCo, momentum update, Contrastive Predictive Coding, mutual information.

7. **Лекция 7 - Introduction to Reinforcement Learning** - базовая постановка обучения с подкреплением и первые алгоритмы.  
   Основные понятия: agent-environment loop, policy, reward, Multi-Armed Bandits, MDP, Markov property, total reward, value function, Q-function, Bellman relations, Cross-Entropy Method, tabular CEM, approximate CEM, continuous action space, Q-learning, approximate Q-learning, Basic Deep Q-Learning, reward formulation.

8. **Лекция 8 - Bellman Equations** - функции ценности, уравнения Беллмана и динамическое программирование в RL.  
   Основные понятия: MDP, reward hypothesis, return, reward discounting, discount factor, reward design, reward scaling, reward shaping, expected objective, backup tree, state-value function, action-value function, Bellman expectation equations, optimal policy, Bellman optimality equations, policy evaluation, policy improvement, generalized policy iteration, policy iteration, value iteration.

9. **Лекция 9 - Model-Free Learning in Reinforcement Learning** - обучение без модели среды и методы на основе траекторий.  
   Основные понятия: model-based vs model-free learning, Monte Carlo estimation, Temporal Difference learning, Q-learning, Monte Carlo vs TD, policy из Q-function, exploration-exploitation tradeoff, approximate Q-learning, MSE formulation, архитектуры approximate Q-learning, Basic Deep Q-Learning.

10. **Лекция 10 - Policy Gradient and REINFORCE** - policy-based методы и градиентная оптимизация политики.  
    Основные понятия: ограничения value-based методов, deterministic и stochastic policies, Cross-Entropy Method как policy-based метод, policy gradient, objective для one-step process, log-derivative trick, REINFORCE, discounted returns, on-policy learning, variance reduction, baselines, advantage function, Actor-Critic, Advantage Actor-Critic, continuous action spaces, entropy regularization, A3C, IMPALA.

11. **Лекция 11 - Reinforcement Learning in Language Modeling and SCST** - применение RL к sequence generation и языковым моделям.  
    Основные понятия: RL-формализм для генерации последовательностей, encoder-decoder architectures, supervised seq2seq learning, attentive translation, sequence generation, distribution shift, неоднозначность правильного ответа, conversation systems, seq2seq как POMDP, policy gradient для sequence generation, cold start problem, training vs inference mismatch, Self-Critical Sequence Training, SCST baseline, image captioning with SCST, discrete GANs, RL fine-tuning.

## Практические задания

- `tasks/task1` - задания по word embeddings и language modeling.
- `tasks/task2` - задания по attention, классической классификации текстов, BERT и переводу.
- `tasks/task3` - задания по reinforcement learning: Cross-Entropy Method, Q-learning и policy gradient.

