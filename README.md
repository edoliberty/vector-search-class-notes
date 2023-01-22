## Vector Search
The course Vector Search is given in Princeton Fall semester 2023 by

* [Edo Liberty](https://scholar.google.com/citations?user=QHS_pZAAAAAJ&hl=en), the Founder and CEO of [Pinecone](www.pinecone.io), the world's leading Vector Database.

* [Matthijs Douze](https://scholar.google.com/citations?user=0eFZtREAAAAJ&hl=en]) the creator and lead maintainer of [FAISS](https://github.com/facebookresearch/faiss) the most popular and advanced open source library for vector search.


The course covers the core concepts, algorithms, and data structures used for modern vector search systems and platforms. An advanced undergraduate or graduate student with some hands-on experience in linear algebra, probability, algorithms, and data structures should be able to follow this course.


### Syllabus

**The class contents below are tentative.**

1. Generalities 
	* embeddings as an information bottleneck _we don't do end-to-end but embeddings in the middle_
	* scalability and typical volumes of data _embeddings are the only way to access large databases_
	* the embedding contract _embedding extractor and embedding indexer agree on the semantic meaning of the distance_ 
	* The vector space model in information retrieval
	* Vector embeddings in machine learning
1. Text embeddings
        * 2-layer word embeddings _word2vec and fastText_ 
	* embedding arithmetic _king + woman - man = queen, already based on similairty search_
	* sentence embeddings _how to train, masked LM_
	* Large Language Models _reasoning as an emerging property of a LM and what happens when training set = the whole web_
1. Image embeddings 
	* Pixel structures of images
	* Traditional CV models 
	* Convolutional Neural Nets
	* Advanced computer vision models
1. Foundations
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
	* k-means
	* k-means++
	* k-means and PCA
1. Product Quantization
	* Hamming encoding
	* Clustering Based Quantization  
1. Graph based indexes
	* Neighborhood graphs
	* HNSW
1. Advanced topics
	* Vectors in Generative AI 
	* Neural quantization 
	* Vector Databases 
1. Project setup
1. Office hours for project work
1. Project Presentations


### Build

On unix-like systems with the bibtex and pdflatex available you should be able to do this:


```
git clone git@github.com:edoliberty/vector-search-class-notes.git
cd vector-search-class-notes
./build
```



### Contribute

These class notes are intended to be used freely by academics anywhere, students and professors alike. Please feel free to contribute in the form of pull requests or opening issues.
