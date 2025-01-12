## GPT-Like LLM made from scratch

Created a model with GPT architecture based on my understanding of [Building a Large Language Model from scratch](https://github.com/rasbt/LLMs-from-scratch/blob/main/)
<br>
<br><a href="https://amzn.to/4fqvn0D"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/cover.jpg?123" width="250px"></a>
<br>

* Created datasets for training, testing and validating the model on textual data from "The Verdict"
* Created a 124M parameter model using:
    * Token and Positional Embeddings
    * Transformers which included Short Circuit, Feed Forward, Multi-Headed Attention and Dropout layers
* Trained the model on my local machine on "the-verdict.txt". The model overfit on this small dataset.
* Loaded the weights for the 124M model open-sources by OpenAI into the model
* Classification Fine Tuning: Created a spam or no-spam classification model by configuring the output layers for the same GPT Model.
* Instruction Fine Tuning: Created a "chatbot" agent by fine tuning the model on a dataset of 1100 instructions.

Note: The model is not very accurate since I could only load the 124M parameter model onto my machine. However, it still functions well as as text completion agent.