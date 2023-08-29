# Long Term Memory in AI - Vector Search and Databases
COS 597A is given at Princeton during the Fall semester 2023 by

* [Edo Liberty](https://scholar.google.com/citations?user=QHS_pZAAAAAJ&hl=en), the Founder and CEO of [Pinecone](https://www.pinecone.io), the world's leading Vector Database.

* [Matthijs Douze](https://scholar.google.com/citations?user=0eFZtREAAAAJ&hl=en]) the architect and main developer of [FAISS](https://github.com/facebookresearch/faiss) the most popular and advanced open source library for vector search.

The course covers the core concepts, algorithms, and data structures used for modern vector search systems and platforms. An advanced undergraduate or graduate student with some hands-on experience in linear algebra, probability, algorithms, and data structures should be able to follow this course.


## Abstract
Long Term Memory is a fundamental capability in the modern AI Stack. At their core, these systems are using Vector search. Vector search is also a fundamental tool for systems that manipulate large collections of media like search engines, knowledge bases, content moderation tools, recommendation systems, etc. As such, the discipline lays at the intersection of Artificial Intelligence and Database Management Systems. This course will cover the scientific foundations and practical implementation of vector search applications, algorithms, and systems. The course will be evaluated with project and in-class presentation. 


## Syllabus

**The class contents below are tentative.**




1. 9/8 - Class 1 - Introduction to Vector Search [Matthijs + Edo]
	* Intro to the course: Topic, Schedule, Project, Grading, ...
	* Embeddings as an information bottleneck. Instead of learning end-to-end, use embeddings as an intermediate representation
	* Advantages: scalability, instant updates, and explainability
	* Typical volumes of data and scalability. Embeddings are the only way to manage / access large databases
	* The embedding contract: the embedding extractor and embedding indexer agree on the meaning of the distance. Separation of concerns.
	* The vector space model in information retrieval
	* Vector embeddings in machine learning
	* Define vector, vector search, ranking, retrieval, recall

	
1. 9/15 - Class 2 - Text embeddings [Matthijs]
	* 2-layer word embeddings. Word2vec and fastText, obtained via a factorization of a co-occurrence matrix. Embedding arithmetic: king + woman - man = queen, (already based on similarity search)
	* Sentence embeddings: How to train, masked LM. Properties of sentence embeddings.
	* Large Language Models: reasoning as an emerging property of a LM. What happens when the training set = the whole web


1. 9/22 - Class 3 - Image embeddings [Matthijs] 
	* Pixel structures of images. Early works on direct pixel indexing
	* Traditional CV models. Global descriptors (GIST). Local descriptors (SIFT and friends)Direct indexing of local descriptors for image matching, local descriptor pooling (Fisher, VLAD)
	* Convolutional Neural Nets. Off-the-shelf models. Trained specifically (contrastive learning, self-supervised learning)
	* Modern Computer Vision models 


1. 9/29 - [Class 4 - Low Dimensional Vector Search](class_notes/Class_04_low_dimensional_vector_search.pdf) [Edo]
	* Vector search problem definition 
	* k-d tree, space partitioning data structures
	* Worst case proof for kd-trees
	* Probabilistic inequalities. Recap of basic inequalities: Markov, Chernoof, Hoeffding
	* Concentration Of Measure phenomena. Orthogonality of random vectors in high dimensions
	* Curse of dimensionality and the failure of space partitioning

1. 10/6 - [Class 5 - Dimensionality Reduction](class_notes/Class_05_dimensionality_reduction.pdf) [Edo] 
	* Singular Value Decomposition (SVD)
	* Applications of the SVD
	* Rank-k approximation in the spectral norm
	* Rank-k approximation in the Frobenius norm
	* Linear regression in the least-squared loss
	* PCA, Optimal squared loss dimension reduction
	* Closest orthogonal matrix
	* Computing the SVD: The power method
	* Random-projection
	* Matrices with normally distributed independent entries
	* Fast Random Projections

1. 10/13 - No Class - Midterm Examination Week

1. 10/20 - No Class - Fall Recess

1. 10/27 - Class 6 - Approximate Nearest Neighbor Search [Edo]
	* Definition of Approximate Nearest Neighbor Search (ANNS)
	* Criteria: Speed / accuracy / memory usage / updateability / index construction time 
	* Definition of Locality Sensitive Hashing and examples
	* The LSH Algorithm
	* LSH Analysis, proof of correctness, and asymptotics

1. 11/3 - Class 7 - Clustering [Edo]
	* Semantic clustering: properties (purity, as aid for annotation)
	* Clustering from a similarity graph (spectral clustering, agglomerative clustering)
	* Vector clustering: mean squared error criterion. Tradeoff with number of clusters
	* Relationship between vector clustering and quantization (OOD extension) 
	* The k-means clustering measure and Lloyd's algorithm
	* Lloyd's optimality conditions
	* Initialization strategies (kmeans++, progressive dimensions with PCA)
	* The inverted file model. Relationship with sparse matrices
	
1. 11/10 - Class 8 - Quantization for lossy vector compression [Matthijs]
	* Vector quantization is a topline (directly optimizes the objective)
	* Binary quantization and hamming comparison 
	* Product quantization. Chunked vector quantization. Optimized vector quantization
	* Additive quantization. Extension of product quantization. Difficulty in training approximations (Residual quantization, CQ, TQ, LSQ, etc.)
	* Cost of coarse quantization vs. inverted list scanning
	
1. 11/17 - Class 9 - Graph based indexes [Guest lecture]
	* Early works: hierarchical k-means 
	* Neighborhood graphs. How to construct them. Nearest Neighbor Descent
	* Greedy search in Neighborhood graphs. That does not work -- need long jumps
	* HNSW. A practical hierarchical graph-based index
	* NSG. Evolving a graph k-NN graph	


1. 11/10 - Class 10 - Computing Hardware and Vector Search [Guest lecture]
	* Computing platform: local vs. service / CPU vs. GPU 
	* efficient implementation of brute force search
	* distance computations for product quantization -- tradeoffs. SIMD implementation
	* Parallelization and distribution -- sharding vs. inverted list distribution
	* Using co-processors (GPUs)
	* Using a hierarchy of memory types (RAM + SSD or RAM + GPU RAM)


1. 11/17 - Class 11- Student project and paper presentations [Edo]

1. 11/24 - No Class - Thanksgiving Recess

1. 12/1 - Class 12 - Student project and paper presentations [Edo]


 
## Selected literature 

* [A fast random sampling algorithm for sparsifying matrices](http://dx.doi.org/10.1007/11830924_26) - Arora, Sanjeev and Hazan, Elad and Kale, Satyen - 2006
* [A Randomized Algorithm for Principal Component Analysis](http://dx.doi.org/10.1137/080736417) - Vladimir Rokhlin and Arthur Szlam and Mark Tygert - 2009
* A search structure based on kd trees for efficient ray tracing - Subramanian, KR and Fussel, DS - 1990
* A Short Proof for Gap Independence of Simultaneous Iteration - Edo Liberty - 2016
* Accelerating Large-Scale Inference with Anisotropic Vector Quantization - Ruiqi Guo and Philip Sun and Erik Lindgren and Quan Geng and David Simcha and Felix Chern and Sanjiv Kumar - 2020
* [Advances in Neural Information Processing Systems 28: Annual Conference on Neural Information Processing Systems 2015, December 7-12, 2015, Montreal, Quebec, Canada](http://papers.nips.cc/book/advances-in-neural-information-processing-systems-28-2015) - 2015
* [An Algorithm for Online K-Means Clustering](https://epubs.siam.org/doi/abs/10.1137/1.9781611974317.7) - Edo Liberty and Ram Sriharsha and Maxim Sviridenko
* An Almost Optimal Unrestricted Fast Johnson-Lindenstrauss Transform - Nir Ailon and Edo Liberty - 2011
* An elementary proof of the Johnson-Lindenstrauss lemma - S. DasGupta and A. Gupta - 1999
* Approximate nearest neighbors and the fast Johnson-Lindenstrauss transform - Nir Ailon and Bernard Chazelle - 2006
* Billion-scale similarity search with GPUs - Jeff Johnson and Matthijs Douze and Herv{\'e} J{\'e}gou - 2017
* Clustering Data Streams: Theory and Practice - Sudipto Guha and Adam Meyerson and Nina Mishra and Rajeev Motwani and Liadan O'Callaghan - 2003
* [DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node](https://proceedings.neurips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf) - Jayaram Subramanya, Suhas and Devvrit, Fnu and Simhadri, Harsha Vardhan and Krishnawamy, Ravishankar and Kadekodi, Rohan - 2019
* Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs - Yu. A. Malkov and D. A. Yashunin - 2018
* [Efficient K-Nearest Neighbor Graph Construction for Generic Similarity Measures](https://doi.org/10.1145/1963405.1963487) - Dong, Wei and Moses, Charikar and Li, Kai - 2011
* Even Simpler Deterministic Matrix Sketching - Edo Liberty - 2022
* Extensions of Lipschitz mappings into a Hilbert space - W. B. Johnson and J. Lindenstrauss - 1984
* Fast Approximate Nearest Neighbor Search With The Navigating Spreading-out Graph - Cong Fu and Chao Xiang and Changxu Wang and Deng Cai - 2018
* [Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions](http://dx.doi.org/10.1137/090771806) - Halko, N. and Martinsson, P. G. and Tropp, J. A. - 2011
* Invertibility of random matrices: norm of the inverse - Mark Rudelson - 2008
* K-means clustering via principal component analysis - Chris H. Q. Ding and Xiaofeng He - 2004
* k-means++: the advantages of careful seeding - David Arthur and Sergei Vassilvitskii - 2007
* Least squares quantization in pcm - Stuart P. Lloyd - 1982
* [LSQ++: Lower Running Time and Higher Recall in Multi-Codebook Quantization](https://doi.org/10.1007/978-3-030-01270-0_30) - Martinez, Julieta and Zakhmi, Shobhit and Hoos, Holger H. and Little, James J. - 2018
* [Multidimensional binary search trees used for associative searching](http://doi.acm.org/10.1145/361002.361007) - Bentley, Jon Louis - 1975
* [Near-Optimal Entrywise Sampling for Data Matrices](https://proceedings.neurips.cc/paper_files/paper/2013/file/6e0721b2c6977135b916ef286bcb49ec-Paper.pdf) - Achlioptas, Dimitris and Karnin, Zohar S and Liberty, Edo - 2013
* Pass Efficient Algorithms for Approximating Large Matrices - Petros Drineas and Ravi Kannan - 2003
* Product Quantization for Nearest Neighbor Search - Jegou, Herve and Douze, Matthijs and Schmid, Cordelia - 2011
* QuickCSG: Arbitrary and faster boolean combinations of n solids - Douze, Matthijs and Franco, Jean-S{\'e}bastien and Raffin, Bruno - 2015
* [Quicker {ADC} : Unlocking the Hidden Potential of Product Quantization With {SIMD](https://doi.org/10.1109%2Ftpami.2019.2952606) - Fabien Andre and Anne-Marie Kermarrec and Nicolas Le Scouarnec - 2021
* [Random Projection Trees and Low Dimensional Manifolds](https://doi.org/10.1145/1374376.1374452) - Dasgupta, Sanjoy and Freund, Yoav - 2008
* [Randomized Algorithms for Low-Rank Matrix Factorizations: Sharp Performance Bounds](http://dx.doi.org/10.1007/s00453-014-9891-7) - Witten, Rafi and Cand\`{e}s, Emmanuel - 2015
* [Randomized Block Krylov Methods for Stronger and Faster Approximate Singular Value Decomposition](http://papers.nips.cc/paper/5735-randomized-block-krylov-methods-for-stronger-and-faster-approximate-singular-value-decomposition) - Cameron Musco and Christopher Musco - 2015
* [Revisiting Additive Quantization](https://api.semanticscholar.org/CorpusID:7340738) - Julieta Martinez and Joris Clement and Holger H. Hoos and J. Little - 2016
* [Sampling from large matrices: An approach through geometric functional analysis](http://doi.acm.org/10.1145/1255443.1255449) - Rudelson, Mark and Vershynin, Roman - 2007
* Similarity estimation techniques from rounding algorithms - Moses Charikar - 2002
* Similarity Search in High Dimensions via Hashing - Aristides Gionis and Piotr Indyk and Rajeev Motwani - 1999
* Simple and Deterministic Matrix Sketching - Edo Liberty - 2012
* Smaller Coresets for k-Median and k-Means Clustering - S. {Har-Peled} and A. Kushal - 2005
* Sparser Johnson-Lindenstrauss transforms - Daniel M. Kane and Jelani Nelson - 2012
* Sparsity Lower Bounds for Dimensionality Reducing Maps - Jelani Nelson and Huy L. Nguyen - 2012
* Spectral Relaxation for K-means Clustering - Hongyuan Zha and Xiaofeng He and Chris H. Q. Ding and Ming Gu and Horst D. Simon - 2001
* Streaming k-means approximation - Nir Ailon and Ragesh Jaiswal and Claire Monteleoni - 2009
* Strong converse for identification via quantum channels - Rudolf Ahlswede and Andreas Winter - 2002
* Transformer Memory as a Differentiable Search Index - Yi Tay and Vinh Q. Tran and Mostafa Dehghani and Jianmo Ni and Dara Bahri and Harsh Mehta and Zhen Qin and Kai Hui and Zhe Zhao and Jai Gupta and Tal Schuster and William W. Cohen and Donald Metzler - 2022
* [Unsupervised Neural Quantization for Compressed-Domain Similarity Search](https://doi.ieeecomputersociety.org/10.1109/ICCV.2019.00313) - S. Morozov and A. Babenko - 2019
* [Worst-Case Analysis for Region and Partial Region Searches in Multidimensional Binary Search Trees and Balanced Quad Trees](https://doi.org/10.1007/BF00263763) - Lee, D. T. and Wong, C. K. - 1977

## Build

On unix-like systems with the bibtex and pdflatex available you should be able to do this:


```
git clone git@github.com:edoliberty/vector-search-class-notes.git
cd vector-search-class-notes
./build
```



## Contribute

These class notes are intended to be used freely by academics anywhere, students and professors alike. Please feel free to contribute in the form of pull requests or opening issues.
