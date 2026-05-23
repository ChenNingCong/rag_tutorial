I am teaching someone who doesn't know RAG and I want to create a tutorial (interactive and runnable one).
There are many simple and easy tutorial online but they are too fragmented. You are reuse these materials. 
Primary resources you should consider, you can download them to learn what they use and replicate some of the steps there (like they may build a vector search for images or books):
1. huggingface's blogs
2. github tutorial for rag with runnable jupyter notebook
Note: in my tutorial, I value two things
1. basic and math reasoning
2. runnable and practice
3. intuition and conciseness
Now I will lye out my current idea for the notebook (you need to generate notebook through python scripts). Feel free to critize or correct me. Note: do not put everything in one notebook split and orgnaize them wisely. Put them in tutorial folder. You'd be better to follow existing notebook's tutorial and setup if possible (when possible, so don't create example by yourself, because the benchmark data might be different, try to reduce from the current dataset)! Add link in your generated notebook to orignal notebook
Since this is complictated. Please generate notebook one by one, run test it to ensure it works. Gemini key is in key.env, an one line string.
Please create an virtual file to run notebook and list all requirements. The notebook code should run all platform so do not use windwos specific path.
When you run the command, notice that this is windows, please use powershell!

1. Basis of retrievel (Note: for this part, you can research other ways to show simple calculation and connection between dot product and tranditional methods, the goal here is to understand the computation of similarity, you can maybe use low dimentional 2D/3D example to show this)
    1.1 old way to do retrieval - tokenization and set intersection (jacard score), with an example. The example should be simple and easy to understand
    1.2 math definition and a python program to compute dot product. Why dot product is defined like this: geometric intuition and linear regression interpreration (weighted sum of feature). Show one example of feature calculation to see how postive and negative get cancelled.
    1.3 Relationship between jacard score and dot product to give more intuition
    1.4 sparse retrieval and bm25, show an example to compute it
2. Semantic retrieval
    1.1 basically follow some famous github/huggingface example (must be simple to setup, no use a complicated vector db). Basically we need to show model's output is a vector (understnad the notation of numpy/torch tensor), and explain the basic of how to call model and get a vector from the input. Must be runnable and cover anything that's important to extract vector, like tokenization, automodel whatever (but don't need to explain model structure) 
    1.2 How to do retrieval: chunking and ranking (chunking is a separate problem so let's assume that input is chunked here), compute and show the scores on small datasets then rank them, to let the user see that similar sentences do get closer score
    1.3 compare tranditional set jacard score method with new method to get scores (compute the result, to show that thery are related)
    1.4 chunking, padding and preprocessing: talk about some common ways to do chunking and preprocessing, show the results on chunk markdown and books (different ways to do this) and run a simple retrieveal to see the results
3. Basic RAG (no reranker)
    Note: this part must talk about embedding and metadata, they are not the same.
    This part is simple, we manually implement a rag pipeline: using the retreival methods we implemented before, then call an API (no fancy langgraph, langchain), just plain language model to do generation. For generation part you need to see the full prompt and how the whole conversation is compiled and assembled together. Note: no agentic, just fix pipeline
4. Concept of embedding models
    Some basic introduction of deep learning stuffs (don't go into too much details)
    In general, you need to note that the basic architecture of BERT model (no need to mention those layers, just mention tokens and vector for each token, then one vector is evaluted). Then for retrieval model it's late interaction. But for reranker this is early interaction. Must show the differnece here
    Basic concept of pretraining and finetuning - why embedding model finetuning is different
5. Reranking
    introduction of 2-stage retrieveal models, this part should be placed after concept of embedding models because it's related how model precess inputs. Show an example how reranking can improve performance of recall (no LLM this part). Just show reranker is better.
6. Evaluation
    how to evaluate RAG: recall, MRF, top-k and RAGAS method, please introduce them one by one and show examples and results
7. Advanced RAG
    corrective RAG, agentic RAG, multiround RAG, basically different way to do prompting and workflow, we can manually implement it.
8. Common failure case of RAG
    a. domain gap
    b. incorrect chunking (forget to pad), or too many duplicate words
    c. question answer gap (hyde)