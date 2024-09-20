# LLM-GenAI
This repo contains projects which include usage of Large Language Models (LLMs) and Generative AI techniques.

In this approch, we have used BioBert, a domain-specific adaptation of BERT for biomedical text, has shown significant improvements in performance on various biomedical NLP tasks, including Named Entity Recognition (NER) in clinical texts.

BioBERT is particularly one of the best models for clinical trials and biomedical data based on the literature available:

BioBERT is pre-trained on large-scale biomedical corpora, which include PubMed abstracts and PMC (PubMed central) full-text articles. This domain-specific pre-training allows BioBERT to capture the unique language, terminology, and contextual understanding of biomedical literature better than general-domain models like BERT. BioBERT significantly outperforms BERT on several biomedical NLP tasks which involve recognizing complex entity names and relationships that are common in clinical trials and medical records.
In a systematic review conducted by researchers from texas and Hong Kong university revealed that pre-trained language models in the biomedical domain, particularly focusing on BioBERT and its related variants (BioALBERT, BioRoBERTa) outperforms ClinicalBERT, BlueBERT and achieves the best performance.
In the same review, researchers showed that PubMedBERT outperformed BioBERT by very little margin in most of the NER tasks except NCBI Disease.
Reference:
1. Lee, Jinhyuk, et al. "BioBERT: a pre-trained biomedical language representation model for biomedical text mining." Bioinformatics 36.4 (2020): 1234-1240. Bioinformatics

2. Benyou Wang, Qianqian Xie, Jiahuan Pei, Zhihong Chen, Prayag Tiwari, Zhao Li, and Jie Fu. 2023. Pre-trained Language Models in Biomedical Domain: A Systematic Survey. ACM Comput. Surv. 56, 3, Article 55 (March 2024), 52 pages. Pre-trained Language Models in Biomedical Domain: A Systematic Survey.

Why Continual Learning?

Prevents Catastrophic Forgetting: Normally, neural networks forget old information when they learn new data. Continual learning helps maintain this old knowledge, which is crucial when the model needs to perform tasks across varied topics without being re-trained from scratch.

Efficiency and Scalability: This method allows for ongoing model updates without revisiting the entire dataset, which saves time and computational resources.

Adaptability: The model can adjust to new changes or learn new tasks or topics while maintaining its performance on previous tasks or topics.


--------------- Detailed Explanation of the Data understanding and Stratified sampling in TASK 1, 2, and 3: -------------------

Data understanding: As a first step, I have gone through data. Here are some important observations:
In the ‘tags’ column the start and end Index for an entity was starting from 1 and going til +2 at the end. 
For example:
			('8:16:chronic_disease,20:32:treatment')
 			( 'portal fibrosis by liver biopsy')

In this example, if we extract out words from the given indexes (8:16) – 'ibrosis '
 and (20:32) – 'iver biopsy '. 

So, I have updated these indexes with correct values.

I figured it out that there are only 4 labels present in the dataset:

‘Unique labels found: {'treatment', 'cancer', 'chronic_disease', 'allergy_name'}
Number of unique label categories: 4’

Train test split using stratification:

I have divided our data into train test split using stratified sampling to have data balance from each category. I faced few challenges in doing stratified sampling because for one data row there might be more than one label present. There were few ways to handle this:

Data Duplication: In this approach I could have considered each label as a single row and duplicated the data. This approach is good for the small dataset but I have enough dataset so, I did not go with this technique. 

Unique-label combination stratification: So, I figured out unique labels combinations present in the whole dataset like:
Unique label combinations found:
('cancer', 'chronic_disease', 'treatment')
('allergy_name',)
('allergy_name', 'treatment')
('allergy_name', 'chronic_disease')
('allergy_name', 'cancer', 'chronic_disease', 'treatment')
('treatment',)
('chronic_disease',)
('allergy_name', 'cancer', 'treatment')
('chronic_disease', 'treatment')
('cancer', 'treatment')
('cancer', 'chronic_disease')
('allergy_name', 'cancer')
('allergy_name', 'chronic_disease', 'treatment')
('cancer',)
Number of unique label combinations: 14

Then I have stratified our dataset based on unique label combinations after merging the combinations as ‘other’ which are having counts ‘<=10’.  



