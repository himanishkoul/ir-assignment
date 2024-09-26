import os
import math
from collections import defaultdict

# Initialize global variables
dictionary = defaultdict(list)
doc_lengths = {}
docID_to_filename = {}
N = 0  # Total number of documents

# Indexing phase
def index_corpus(corpus_path):
    global N
    for docID, filename in enumerate(os.listdir(corpus_path)):
        docID_to_filename[docID] = filename  # Store filename for later use
        N += 1
        with open(os.path.join(corpus_path, filename), 'r', encoding='utf-8') as file:
            content = file.read().lower().split()  # Basic tokenization
            term_freqs = defaultdict(int)
            # Calculate term frequencies
            for term in content:
                term_freqs[term] += 1
            
            # Update dictionary and postings
            for term, tf in term_freqs.items():
                dictionary[term].append((docID, tf))
                
            # Calculate and store document length
            length = math.sqrt(sum((1 + math.log10(tf)) ** 2 for tf in term_freqs.values()))
            doc_lengths[docID] = length

# Query phase
def process_query(query):
    query_terms = query.lower().split()  
    query_weights = defaultdict(float)
    
    # Calculate tf-idf for query with boosted weights
    query_length = 0
    for term in query_terms:
        tf = query_terms.count(term)
        df = len(dictionary[term]) if term in dictionary else 0
        idf = math.log10((N + 1) / (df + 0.5))  # Adjusted IDF for smoothing
        tf_idf = (1 + math.log10(tf)) * idf  # Log-normalized tf-idf
        query_weights[term] = tf_idf
        query_length += tf_idf ** 2  # Calculate query length for normalization
    
    query_length = math.sqrt(query_length)  # Normalize the query vector
    
    # Rank documents by cosine similarity
    scores = defaultdict(float)
    for term, query_weight in query_weights.items():
        if term in dictionary:
            for docID, tf in dictionary[term]:
                # Use log-normalized tf-idf for document weight
                doc_weight = (1 + math.log10(tf)) * math.log10((N + 1) / (len(dictionary[term]) + 0.5))
                scores[docID] += (query_weight * doc_weight)
    
    for docID in scores:
        if doc_lengths[docID] > 0 and query_length > 0:
            scores[docID] /= (doc_lengths[docID] * query_length)
    
    for docID in scores:
        scores[docID] = 2 / (1 + math.exp(-10 * scores[docID])) - 1
    
    # Sort by score and then by docID
    ranked_docs = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    
    # Output top 10 results with actual filenames
    return [(docID_to_filename[docID], score) for docID, score in ranked_docs[:10]]

# Main Function
def main():
    corpus_path = r'C:\\Users\\Himanish\\Desktop\\IR Assignment 2\\Corpus'  # Use raw string
    index_corpus(corpus_path)
    
    # Ask user for input query
    query = input("Enter your search query: ")
    
    results = process_query(query)
    
    # Display the results
    print("\nTop 10 relevant documents:")
    for doc, score in results:
        print(f"{doc}: {score:.15f}") 

if __name__ == "__main__":
    main()
