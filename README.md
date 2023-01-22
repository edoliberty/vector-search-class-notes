# Vector Search
The course Vector Search is given in Princeton Fall semester 2023 by

* [Edo Liberty](https://scholar.google.com/citations?user=QHS_pZAAAAAJ&hl=en), the Founder and CEO of [Pinecone](www.pinecone.io), the world's leading Vector Database.

* [Matthijs Douze](https://scholar.google.com/citations?user=0eFZtREAAAAJ&hl=en]) the creator and lead maintainer of [FAISS](https://github.com/facebookresearch/faiss) the most popular and advanced open source library for vector search.


The course covers the core concepts, algorithms, and data structures used for modern vector search systems and platforms. An advanced undergraduate or graduate student with some hands-on experience in linear algebra, probability, algorithms, and data structures should be able to follow this course.


## Syllabus

**The class contents below are tentative.**

1. Generalities 
	* Embeddings as an information bottleneck. Instead of learning end-to-end, use embeddings as an intermediate representation.
	* Typical volumes of data and scalability. Embeddings are the only way to manage / access large databases
	* The embedding contract. The embedding extractor and embedding indexer agree on the meaning of the distance
	* The vector space model in information retrieval
	* Vector embeddings in machine learning
	
1. Text embeddings
	* 2-layer word embeddings. Word2vec and fastText, obtained via a factorization of a co-ocurrence matrix. Embedding arithmetic: king + woman - man = queen, which is already based on similairty search
	* Sentence embeddings: How to train, masked LM. Properties of sentence embeddings
	* Large Language Models: reasoning as an emerging property of a LM and what happens when training set = the whole web

1. Image embeddings 
	* Pixel structures of images. Early works on pixel indexing
	* Traditional CV models. Global descriptors (GIST). Local descriptors (SIFT and friends), direct indexing for image matching, pooling (Fisher, VLAD)
	* Convolutional Neural Nets. Off-the-shelf models. Trained specifically (constrastive learning, self-supervised learning). 
	* Advanced computer vision models 

1. Practical indexing
	* How an index works: basic (search, add) and optional functionalities (snapshot, incremental add, remove)
	* k-NN search vs. range search 
	* Criteria: Speed / accuracy / memory usage / updateability / index construction time 
	* Computing platform: local / service / CPU / GPU 

1. Mathematical foundations
	* Recap of relevant Linear Algebra
	* Vector operations and notation
	* Probabilistic inequalities	

1. Low dimensional vector search 
	* k-d tree
	* Concentration phenomena
	* Curse of dimensionality

1. Dimensionality Reduction I
	* Principal Components Analysis 
	* Fast PCA algorithms 

1. Dimensionality Reduction II
	* Random Projections
	* Fast Random Projections	

1. Locality Sensitive Hashing
	* Definition of ANN
	* LSH Algorithm
	* LSH Analysis
	
1. Clustering
	* Semantic clustering: properties (purity, as aid for annotation)
	* Clustering from a similarity graph (spectral clustering)
	* Vector clustering: mean squared error criterion. Tradeoff with number of clusters
	* Relationship between vector clustering and quantization (OOD extension) 

1. Vector quantization 
	* Lloyd's optimality conditions. 
	* The k-means algorithm 
	* Exact k-means 
	* Initialization strategies (kmeans++, progressive dimensions with PCA)
	
1. Quantization for lossy vector compression
	* Vector quantization is a topline (directly optimizes the objective)
	* Binary quantization and hamming comparison 
	* Product quantization. Chunked vector quantization. Optimized vector quantization
	* Additive quantization. Extension of product quantization. Difficult to train: approximations (Residual quantization, CQ, TQ, LSQ, etc.)

1. Non-exhaustive search 
	* The bag-of-visual-words inspiration. 
	* Voronoi diagram with search buckets.
	* The inverted file model. 
	* Cost of coarse quantization vs. inverted list scanning. 

1. Graph based indexes
	* Early works: hierarchical k-means 
	* Neighborhood graphs. How to construct them. NNdescent. 
	* Greedy search in Neighborhood graphs. That does not work -- need shortcuts. 
	* HNSW. A practical hierarchical graph-based index
	
1. Advanced topics
	* Vectors in Generative AI 
	* Neural quantization 
	* Vector Databases 
	
1. Project setup

1. Office hours for project work

1. Project Presentations


## Build

On unix-like systems with the bibtex and pdflatex available you should be able to do this:


```
git clone git@github.com:edoliberty/vector-search-class-notes.git
cd vector-search-class-notes
./build
```



## Contribute

These class notes are intended to be used freely by academics anywhere, students and professors alike. Please feel free to contribute in the form of pull requests or opening issues.
