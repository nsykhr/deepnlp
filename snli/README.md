### 1. Task description
#### 1. 3-way classification: neutral, entailment, contradiction
Dataset: SNLI. Metrics: accuracy, f1-score.
#### 2. Meaningful sentence representation learning
Dataset: Quora Question Pairs (used for testing only). Metric: ROC-AUC over sentence embedding dot products.

### 2. Baselines
Here: https://nlp.stanford.edu/projects/snli/

The majority of papers use Siamese neural networks with a classification
head on top of the encoder. In the last few years, only Transformer-based
networks are used for the task because it requires a deep understanding
of language. The most successful ones are those who fine-tune models pretrained
on metric learning tasks with the goal of encoding sentence semantics and structure
in one dense vector.

### 3. Brief summary
- the models and the training loop were not overly hard to write;
the only thing I could call slightly tricky is AdamW initialization
with different learning rates and weight decay for different layers of my network
- creating custom Dataset classes was definitely harder and took more time,
especially because some parts had to be reworked (e. g., at first I padded
my sequences manually before I discovered that tokenizers from `transformers`
can do it much more efficiently; another thing that was absent in the
early drafts of the code is attention masks for padding)
- experiments definitely took the most time since I had to re-start training
dozens of times to see how loss behaves depending on different hyper-parameters,
Transformer models under the hood, data reading approaches (random batches vs. buckets)
- among the erroneous decisions that were made was simultaneous fine-tuning of two
Transformer encoders for the premises and for the hypotheses — but it was corrected
in time by @BobaZooba (the project mentor)
- metrics-wise, the final result is not awe-inspiring (some of the reasons why I think
it could be so are listed in the `evaluation.ipynb` notebook)
- in the end, 4 models were trained. All of them follow the standard BiEncoder architecture
(Siamese network). Additionally, there is working code to create a CrossEncoder model
that encodes the premise and the hypothesis simulaneously, relying on special tokens for
the model to differentiate them. In the table below are the obtained values of accuracy.

 | | ALBERT | ELECTRA
 | --- | --- | --- |
 *batching* | 0.8065 | **0.8262**
 *bucketing* | 0.8139 | 0.8194