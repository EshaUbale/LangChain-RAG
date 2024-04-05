#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install sentence-transformers langchain huggingface_hub faiss-cpu langchainhub langchain_community


# In[2]:


import argparse
import getpass
import numpy as np
import os
import pandas as pd
import random
from time import time
from time import perf_counter
import torch
from tqdm import tqdm

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains.question_answering import load_qa_chain
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from datasets import Dataset
from transformers import AutoTokenizer, AutoModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
#from transformers import pipeline


# In[3]:


#HUGGINGFACEHUB_API_TOKEN = getpass.getpass("Enter your HF API Key:\n\n")
HUGGINGFACEHUB_API_TOKEN = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
# from huggingface_hub import login
# login(HUGGINGFACEHUB_API_TOKEN)
#LANGCHAIN_API_TOKEN = "ls__3e13508fcd224822ba400997518f265b"
#os.environ["LANGCHAIN_TRACING_V2"] = "false"
#os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
#os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_TOKEN
#os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()

seed_val = 1234
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


# In[4]:


def build_simple_qa_chain():
    # "google/flan-t5-base" works for the free account
    qa_model_llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
            task="text-generation",
            model_kwargs={"temperature": 1, "max_length": 256}
        )
    
    # template = """Question: {question}
    
    # Answer: """

    # prompt = PromptTemplate(
    #         template=template,
    #         input_variables=['question']
    # )
    
    #template = "Q: {question} A:"
    #template = "Answer the following question:\n\n{question}"
    template = "Question: {question}?\nAnswer:"
    prompt = PromptTemplate.from_template(template)
    llm_chain = LLMChain(prompt=prompt, llm=qa_model_llm)

    return llm_chain


# In[5]:


def test_no_rag(questions_path, output_path):
    print("Using questions file=", questions_path, " and output_path=", output_path)

    questions = pd.read_csv(questions_path)
    #print(questions.head())
    question = questions["question"][0]
    #print(question)

    tick = perf_counter()
    qa_chain = build_simple_qa_chain()
    print(f'Time span for building QA chain: {perf_counter() - tick}')
    
    tick = perf_counter()
    ans = qa_chain.invoke(question)
    print(f'Time span for single query: {perf_counter() - tick}')
    
    print("-----")
    print(ans['question'])
    print(ans['text'])
    print("-----")

    tick = perf_counter()
    questions_list = []
    results = []
    for i in range(len(questions)):
        q = questions["question"][i]
        try:
            ans = qa_chain.invoke(q)
            questions_list.append(q)
            results.append(ans['text'])
        except Exception as error:
            print("An error occurred:", type(error).__name__, "–", error)
            questions_list.append(q)
            results.append("Error")

    print("")
    print(f'Time span for all questions: {perf_counter() - tick}')
    print("")
    
    dict = {'question': questions_list, 'answer': results}
    df_pred = pd.DataFrame(dict)
    df_pred.head()
    df_pred.to_csv(output_path, index_label="#")
    
    return


# In[6]:


def get_vector_store(passages_path, chunk_size, chunk_overlap):
    print("Loading passages from", passages_path)
    # Create a document loader for fifa_countries_audience.csv
    reader = CSVLoader(file_path=passages_path, encoding="utf-8", csv_args={'delimiter': ','})
    
    # Step 1 : Load the document
    data = reader.load()
    #print(data[0])
    
    # Step 2: Chunk the documents. Breaks down the loaded documents into smaller, more manageable pieces to facilitate efficient retrieval.
    separator = "\n"
    # Create an instance of the splitter class
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=separator
        )
    # Split the document and print the chunks
    text_chunks = splitter.split_documents(data)

    print("chunk_size =", chunk_size, " chunk_overlap =", chunk_overlap)
    print("# text chunks = ", len(text_chunks))
    #print(text_chunks[0])
    #print(len(text_chunks[0].page_content))
    #print(text_chunks[0].metadata)
    
    # Step 3: Embed Documents. Converts the textual chunks into vector representations, making them searchable within the system.
    # Initializing embeddings using Hugging Face models
    embeddings = HuggingFaceEmbeddings()
    # Creating a FAISS (Facebook's AI Similarity Search) vector store from the document chunks and embeddings
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    
    # Step 4: Store Embeddings. Safely stores the generated embeddings alongside their textual counterparts for future retrieval.
    #DB_FAISS_PATH = "db_faiss"
    #vector_store.save_local(DB_FAISS_PATH)
    
    #embeddings = HuggingFaceEmbeddings()
    #new_db = FAISS.load_local(DB_FAISS_PATH, embeddings)

    return vector_store


# In[7]:


def build_qa_chain(vectorstore, top_k):
    #docs = vectorstore.similarity_search(query)
    # Make retriever
    #retriever = vectorstore.as_retriever()
    # Specifying top k
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    
    #retrieved_docs = retriever.invoke(question)
    #print(len(retrieved_docs))
    #print(retrieved_docs[0].page_content)
    #for doc in retrieved_docs:
    #    print(doc.page_content)

    # "google/flan-t5-base" works for the free account
    qa_model_llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
            task="text-generation",
            model_kwargs={"temperature": 1, "max_length": 256}
        )

    prompt = hub.pull("rlm/rag-prompt")
    # example_messages = prompt.invoke(
    #     {"context": "filler context", "question": "filler question"}
    # ).to_messages()
    # print(example_messages[0].content)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # rag_chain = (
    #     {"context": retriever | format_docs, "question": RunnablePassthrough()}
    #     | prompt
    #     | qa_model_llm
    #     | StrOutputParser()
    # )

    # print(f"Question: {question}")
    # print("Answer:")
    # for chunk in rag_chain.stream(question):
    #     print(chunk, end="", flush=True)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | qa_model_llm
        | StrOutputParser()
    )
    
    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    return rag_chain_with_source
    


# In[8]:


def test_rag_langchain(questions_path, output_path, passages_path):
    print("Using RAG with LangChain with questions file=", questions_path, " and output_path=", output_path)

    questions = pd.read_csv(questions_path)
    #print(questions.head())
    question = questions["question"][0]
    #print(question)

    tick = perf_counter()
    vectorstore = get_vector_store(passages_path=passages_path, chunk_size=50, chunk_overlap=0)
    print(f'Time span for building vector store: {perf_counter() - tick}')

    tick = perf_counter()
    qa_chain = build_qa_chain(vectorstore=vectorstore, top_k=6)
    print(f'Time span for building QA chain: {perf_counter() - tick}')
    
    tick = perf_counter()
    ans = qa_chain.invoke(question)
    print(f'Time span for single query: {perf_counter() - tick}')
    
    print("-----")
    print(ans['question'])
    print(ans['answer'])
    # print("Texts")
    # for i in range(len(ans['context'])):
    #     print( ans['context'][i].page_content.removeprefix("context: ") )
    print("-----")

    tick = perf_counter()
    questions_list = []
    results = []
    sources1 = []
    sources2 = []
    sources3 = []
    #for q in tqdm(questions["question"]):
    for i in range(len(questions)):
        q = questions["question"][i]
        try:
            ans = qa_chain.invoke(q)
            questions_list.append(q)
            results.append(ans['answer'])
            sources1.append( ans['context'][0].page_content.removeprefix("context: ") )
            sources2.append( ans['context'][1].page_content.removeprefix("context: ") )
            sources3.append( ans['context'][2].page_content.removeprefix("context: ") )
        except Exception as error:
            print("An error occurred:", type(error).__name__, "–", error)
            questions_list.append(q)
            results.append("Error")
            sources1.append("Error")
            sources2.append("Error")
            sources3.append("Error")

    print("")
    print(f'Time span for all questions: {perf_counter() - tick}')
    print("")
    
    dict = {'question': questions_list, 'answer': results, 'source1': sources1, 'source2': sources2, 'source3': sources3}
    df_pred = pd.DataFrame(dict)
    df_pred.head()
    df_pred.to_csv(output_path, index_label="#")
    #df_pred.to_csv(output_path, index=None)

    #for chunk in rag_chain_with_source.stream(question):
    #    print(chunk, end="", flush=True)

    #qa_chain = load_qa_chain(qa_model_llm, chain_type="stuff")
    #search_results = vectorstore.similarity_search(question)
    #answers = qa_chain.invoke(input_documents=search_results, question=question)
    #print(f"Question: {question}")
    #print(f"Answer: {answers}")
    
    return


# In[9]:


# def concatenate_text(examples):
#     return {
#         "text": examples["title"]
#         + " \n "
#         + examples["context"]
#     }

def concatenate_text(examples):
    return {
        "text": examples["context"]
    }

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    # return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# Perform CLS pooling on our model’s outputs, where we simply collect the last hidden state for the special [CLS] token
def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]
    
def get_embeddings(text_list, tokenizer, model, device):
    # Tokenize sentences
    # Tokenize each document in passages.csv
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    # Compute token embeddings
    model_output = model(**encoded_input)
    return(mean_pooling(model_output, encoded_input['attention_mask']))
    #return cls_pooling(model_output)

def create_prompt(docs, question):
    return "Context: " + '\n\n'.join(docs) + "\n\nQuestion: " + question + "\n\nAnswer:"

def run_query(tokenizer, model, device, context_docs, question):
    input_text =  create_prompt(context_docs, question)
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    
    outputs = model.generate(input_ids)
    #print(outputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True )


# In[17]:


import numpy as np
from scipy.spatial.distance import cosine, euclidean
from sentence_transformers.util import cos_sim, dot_score

# Use numpy for speed

def retrieve_by_cosine_similarity(embeddings_dataset, embeddings_column, query_embedding, top_k):
    temp_list = []
    for i in range(len(embeddings_dataset)):
        passage_embedding = embeddings_dataset[embeddings_column][i]
        cosine_sim = np.round( np.dot(query_embedding, passage_embedding) / (np.linalg.norm(query_embedding)*np.linalg.norm(passage_embedding)), 4)
        #cosine_sim = np.round( (1 - cosine(question_embedding, passage_embedding)), 4)
        #cosine_sim = cos_sim(question_embedding, passage_embedding)
        temp_list.append( {"index": i, "score": cosine_sim} )
        
    newlist = sorted( temp_list, key=lambda d: d['score'], reverse=True )[:top_k]

    #print(newlist)
    return [d['score'] for d in newlist], embeddings_dataset[[d['index'] for d in newlist]]

def retrieve_by_dot_product(embeddings_dataset, embeddings_column, query_embedding, top_k):
    temp_list = []
    for i in range(len(embeddings_dataset)):
        passage_embedding = embeddings_dataset[embeddings_column][i]
        dot_p = np.round( np.dot(query_embedding, passage_embedding), 4)
        #dot_p = dot_score(query_embedding, passage_embedding)
        temp_list.append( {"index": i, "score": dot_p} )
        
    newlist = sorted( temp_list, key=lambda d: d['score'], reverse=True )[:top_k]

    #print(newlist)
    return [d['score'] for d in newlist], embeddings_dataset[[d['index'] for d in newlist]]

def retrieve_by_euclidean(embeddings_dataset, embeddings_column, query_embedding, top_k):
    temp_list = []
    for i in range(len(embeddings_dataset)):
        passage_embedding = embeddings_dataset[embeddings_column][i]
        #e = np.round( euclidean(query_embedding, passage_embedding), 4)
        e = np.round( np.linalg.norm(query_embedding - passage_embedding), 4)
        temp_list.append( {"index": i, "score": e} )
        
    newlist = sorted( temp_list, key=lambda d: d['score'], reverse=False )[:top_k]

    #print(newlist)
    return [d['score'] for d in newlist], embeddings_dataset[[d['index'] for d in newlist]]

def print_results(scores, samples):
    samples_df = pd.DataFrame.from_dict(samples)
    samples_df["scores"] = scores
    # samples_df.sort_values("scores", ascending=True, inplace=True)
    print(samples_df[["#", "text", "scores"]])



# In[18]:


def test_rag(questions_path, output_path, passages_path):
    print("Using RAG with questions file=", questions_path, " and output_path=", output_path)

    questions = pd.read_csv(questions_path)
    #print(questions.head())
    #question = questions["question"][8]
    question = "Which presidential administration developed Safe Harbor policy?"
    #print(question)

    tick = perf_counter()

    passages_df = pd.read_csv('passages.csv')
    #print("\n")
    #print(passages_df.head())
    
    passages_dataset = Dataset.from_pandas(passages_df)
    #print("\n")
    #print(passages_dataset)

    print("Generating context texts from passages")
    passages_dataset = passages_dataset.map(concatenate_text)
    
    # Load model from HuggingFace Hub
    # Use the sentence-transformers/roberta-base-nli-stsb-mean-tokens model and its corresponding tokenizer available on Hugging Face.
    model_ckpt = 'sentence-transformers/roberta-base-nli-stsb-mean-tokens'
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModel.from_pretrained(model_ckpt)
    #device = "cpu"
    model.to(device)

    # apply get_embeddings() function to "text" column in each row and store in "embeddings" column as numpy array
    print("Generating embeddings")
    embeddings_dataset = passages_dataset.map(
        lambda x: {"embeddings": get_embeddings(x["text"], tokenizer, model, device).detach().cpu().numpy()[0]}
    )    
    
    # Creating a FAISS index
    embeddings_dataset.add_faiss_index(column="embeddings")
    # import faiss
    # faiss_num_dim = 768
    # faiss_num_links = 128
    # index = faiss.IndexHNSWFlat(faiss_num_dim, faiss_num_links, faiss.METRIC_INNER_PRODUCT)
    # embeddings_dataset.add_faiss_index("embeddings", custom_index=index)
    
    # embeddings_dataset.save_faiss_index('embeddings', 'esha_index.faiss')
    # embeddings_dataset.drop_index('embeddings')
    # embeddings_dataset.save_to_disk(dataset_path='esha_dataset')
    # embeddings_dataset.load_faiss_index('embeddings', 'esha_index.faiss')
    # embeddings_dataset.add_elasticsearch_index("text", host="localhost", port="9200", es_index_name="hf_val_context")
    
    print(f'Time span for building vector store: {perf_counter() - tick}')

    qa_checkpoint = "google/flan-t5-base"
    qa_tokenizer = T5Tokenizer.from_pretrained(qa_checkpoint)
    qa_model = T5ForConditionalGeneration.from_pretrained(qa_checkpoint, device_map="auto")
    qa_model.to(device)

    k = 10 # top k passages
    tick = perf_counter()
    question_embedding = get_embeddings([question], tokenizer, model, device).cpu().detach().numpy()
    scores, samples = embeddings_dataset.get_nearest_examples("embeddings", question_embedding, k=k)
    ans = run_query(qa_tokenizer, qa_model, device, samples['text'], question)
    print(f'Time span for single query: {perf_counter() - tick}')
    
    print("-----")
    print(question)
    print(ans)
    # print("Texts")
    # for i in range(len(samples['text'])):
    #     print( samples['text'][i] )
    print("-----")

    compare_search = False

    if (compare_search):
        print("\n Comparing Search Methods for ")
        print("Question:", question)
        print("\nSearch by faiss")
        tick = perf_counter()
        scores, samples = embeddings_dataset.get_nearest_examples("embeddings", question_embedding, k=k)
        print(f'Time span for search: {perf_counter() - tick}')
        ans = run_query(qa_tokenizer, qa_model, device, samples['text'], question)
        print_results(scores, samples)
        print("Answer using top_k passages:", ans)
        
        print("\nSearch by cosine similarity")
        tick = perf_counter()
        scores, samples = retrieve_by_cosine_similarity(embeddings_dataset, "embeddings", question_embedding[0], top_k=k)
        print(f'Time span for search: {perf_counter() - tick}')
        print_results(scores, samples)
        print("Answer using top_k passages:", ans)
        
        print("\nSearch by dot product")
        tick = perf_counter()
        scores, samples = retrieve_by_dot_product(embeddings_dataset, "embeddings", question_embedding[0], top_k=k)
        print(f'Time span for search: {perf_counter() - tick}')
        print_results(scores, samples)
        print("Answer using top_k passages:", ans)
        
        print("\nSearch by euclidean")
        tick = perf_counter()
        scores, samples = retrieve_by_euclidean(embeddings_dataset, "embeddings", question_embedding[0], top_k=k)
        print(f'Time span for search: {perf_counter() - tick}')
        print_results(scores, samples)
        print("Answer using top_k passages:", ans)
        return
        
    
    tick = perf_counter()
    questions_list = []
    results = []
    sources1 = []
    sources2 = []
    sources3 = []
    #for q in tqdm(questions["question"]):
    for i in range(len(questions)):
        q = questions["question"][i]
        try:
            question_embedding = get_embeddings([q], tokenizer, model, device).cpu().detach().numpy()
            scores, samples = embeddings_dataset.get_nearest_examples("embeddings", question_embedding, k=k)
            #scores, samples = retrieve_by_cosine_similarity(embeddings_dataset, "embeddings", question_embedding[0], top_k=k)
            #scores, samples = retrieve_by_dot_product(embeddings_dataset, "embeddings", question_embedding[0], top_k=k)
            #scores, samples = retrieve_by_euclidean(embeddings_dataset, "embeddings", question_embedding[0], top_k=k)
            ans = run_query(qa_tokenizer, qa_model, device, samples['text'], q)
            questions_list.append(q)
            results.append(ans)
            sources1.append( samples['text'][0] )
            sources2.append( samples['text'][1] )
            sources3.append( samples['text'][2] )
        except Exception as error:
            print("An error occurred:", type(error).__name__, "–", error)
            questions_list.append(q)
            results.append("Error")
            sources1.append("Error")
            sources2.append("Error")
            sources3.append("Error")

    print("")
    print(f'Time span for all questions: {perf_counter() - tick}')
    print("")
    
    dict = {'question': questions_list, 'answer': results, 'source1': sources1, 'source2': sources2, 'source3': sources3}
    df_pred = pd.DataFrame(dict)
    df_pred.head()
    df_pred.to_csv(output_path, index_label="#")
    
    return


# In[12]:


def is_path_exists(parser, arg):
    if not os.path.exists(arg):
        parser.error('The path {} does not exist!'.format(arg))
    else:
        # File exists so return the path
        return arg

def is_valid_file(parser, arg):
    if not os.path.isfile(arg):
        parser.error('The file {} does not exist!'.format(arg))
    else:
        # File exists so return the filename
        return arg

def is_valid_directory(parser, arg):
    if not os.path.isdir(arg):
        parser.error('The directory {} does not exist!'.format(arg))
    else:
        # File exists so return the directory
        return arg
 


# In[13]:


def main():
    parser = argparse.ArgumentParser(description='Homework CLI')
    parser.add_argument("--questions", default=None, type=str, required=True, help="path to the questions file")
    parser.add_argument("--output", default=None, type=str, required=True, help="path for saving the predictions")
    parser.add_argument("--rag", default=False, action='store_true', help="indicator to use RAG")
    parser.add_argument("--langchain", default=False, action='store_true', help="indicator to use RAG")
    parser.add_argument("--passages", default=None, type=str, help="path for reading the passages")
    
    # python main.py --questions questions.csv --output predictions_no_rag.csv
    # python main.py --questions questions.csv --rag --langchain --passages passages.csv --output predictions_rag_langchain.csv
    # python main.py --questions questions.csv --rag --passages passages.csv --output predictions_rag.csv

    #args = parser.parse_args(['--questions', 'questions.csv', '--output', 'predictions_no_rag.csv'])
    #args = parser.parse_args(['--questions', 'questions.csv', '--rag', '--langchain', '--passages', 'passages.csv', '--output', 'predictions_rag_langchain.csv'])
    #args = parser.parse_args(['--questions', 'questions.csv', '--rag', '--passages', 'passages.csv', '--output', 'predictions_rag.csv'])
    #args = parser.parse_args(['--questions', 'val_questions.csv', '--rag', '--passages', 'passages.csv', '--output', 'pred_val_no_langchain.csv'])
    args = parser.parse_args()

    is_valid_file(parser, args.questions)
    is_path_exists(parser, args.questions)
    #is_valid_file(parser, args.output)

    if args.rag:
        is_valid_file(parser, args.passages)
        is_path_exists(parser, args.passages)
        
        if args.langchain:
            test_rag_langchain(args.questions, args.output, args.passages)
        else:
            test_rag(args.questions, args.output, args.passages)
    else:
        test_no_rag(args.questions, args.output)
    
 


# In[19]:


if __name__ == "__main__":
    main()
    


# In[ ]:




