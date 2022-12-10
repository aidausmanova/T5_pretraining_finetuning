# T5_pretraining_finetuning

This project explores new T5 model by Hugging Face (https://huggingface.co/docs/transformers/model_doc/t5).

### ConceptNet
You can download ConceptNet assertions (https://github.com/commonsense/conceptnet5/wiki/Downloads) and save in "data" folder.

Pre-processing steps:
1. Filter out all non-english assertions
2. Randomly select 5 triplets from each Subject
3. Randomly mask 2 words in each sentence

### Pre-training
The pre-training is done following the example of T5 (https://huggingface.co/docs/transformers/model_doc/t5#training) on Masked Language Modeling (MLM) using previously pre-processed ConceptNet triplets.

### Fine-tuning
In the end T5 model is fine-tuned on TellMeWhy dataset (https://stonybrooknlp.github.io/tellmewhy/) for Q&A task.

