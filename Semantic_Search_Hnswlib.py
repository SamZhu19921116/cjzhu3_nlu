from sentence_transformers import SentenceTransformer, util
import os
import pickle
import time
import hnswlib

model_name = '/home/cjzhu3/NLU_CJZHU3/SBert/distiluse-base-multilingual-cased-v1'
model = SentenceTransformer(model_name)
dataset_path = "/home/cjzhu3/NLU_CJZHU3/data/ZhongAnXiaoDai_Corpus_query.txt"
max_corpus_size = 100000
embedding_cache_path = '/home/cjzhu3/NLU_CJZHU3/model/retrieval/faiss_sbert_embeddings.pkl'
embedding_size = 512 #Size of embeddings
top_k_hits = 10 #Output k hits

#Check if embedding cache path exists
if not os.path.exists(embedding_cache_path):
    # Get all unique sentences from the file
    corpus_sentences = set()
    with open(dataset_path, encoding='utf8') as fIn:
        for line in fIn:
            corpus_sentences.add(line.strip())
            if len(corpus_sentences) >= max_corpus_size:
                break

    corpus_sentences = list(corpus_sentences)
    print("Encode the corpus. This might take a while")
    corpus_embeddings = model.encode(corpus_sentences, show_progress_bar=True, convert_to_numpy=True)

    print("Store file on disc")
    with open(embedding_cache_path, "wb") as fOut:
        pickle.dump({'sentences': corpus_sentences, 'embeddings': corpus_embeddings}, fOut)
else:
    print("Load pre-computed embeddings from disc")
    with open(embedding_cache_path, "rb") as fIn:
        cache_data = pickle.load(fIn)
        corpus_sentences = cache_data['sentences']
        corpus_embeddings = cache_data['embeddings']

#Defining our hnswlib index
index_path = "/home/cjzhu3/NLU_CJZHU3/model/retrieval/hnswlib_sbert_embeddings.index"
#We use Inner Product (dot-product) as Index. We will normalize our vectors to unit length, then is Inner Product equal to cosine similarity
index = hnswlib.Index(space = 'cosine', dim = embedding_size)

if os.path.exists(index_path):
    print("Loading index...")
    index.load_index(index_path)
else:
    ### Create the HNSWLIB index
    print("Start creating HNSWLIB index")
    index.init_index(max_elements = len(corpus_embeddings), ef_construction = 400, M = 64)
    # Then we train the index to find a suitable clustering
    print(len(corpus_embeddings))
    print(len(corpus_embeddings[0]))
    #index.add_items(corpus_embeddings)
    #print(list(range(len(corpus_embeddings))))
    index.add_items(corpus_embeddings, list(range(len(corpus_embeddings))))

    print("Saving index to:", index_path)
    index.save_index(index_path)
# Controlling the recall by setting ef:
index.set_ef(50)  # ef should always be > top_k_hits

######### Search in the index ###########

print("Corpus loaded with {} sentences / embeddings".format(len(corpus_sentences)))

while True:
    inp_question = input("Please enter a question: ")

    start_time = time.time()
    question_embedding = model.encode(inp_question)

    #We use hnswlib knn_query method to find the top_k_hits
    corpus_ids, distances = index.knn_query(question_embedding, k=top_k_hits)

    # We extract corpus ids and scores for the first query
    hits = [{'corpus_id': id, 'score': 1-score} for id, score in zip(corpus_ids[0], distances[0])]
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)
    end_time = time.time()

    print("Input question:", inp_question)
    print("Results (after {:.3f} seconds):".format(end_time-start_time))
    for hit in hits[0:top_k_hits]:
        print("\t{:.3f}\t{}".format(hit['score'], corpus_sentences[hit['corpus_id']]))

    # Approximate Nearest Neighbor (ANN) is not exact, it might miss entries with high cosine similarity
    # Here, we compute the recall of ANN compared to the exact results
    correct_hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k_hits)[0]
    correct_hits_ids = set([hit['corpus_id'] for hit in correct_hits])

    ann_corpus_ids = set([hit['corpus_id'] for hit in hits])
    if len(ann_corpus_ids) != len(correct_hits_ids):
        print("Approximate Nearest Neighbor returned a different number of results than expected")

    recall = len(ann_corpus_ids.intersection(correct_hits_ids)) / len(correct_hits_ids)
    print("\nApproximate Nearest Neighbor Recall@{}: {:.2f}".format(top_k_hits, recall * 100))

    if recall < 1:
        print("Missing results:")
        for hit in correct_hits[0:top_k_hits]:
            if hit['corpus_id'] not in ann_corpus_ids:
                print("\t{:.3f}\t{}".format(hit['score'], corpus_sentences[hit['corpus_id']]))
    print("\n\n========\n")
