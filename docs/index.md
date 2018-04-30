# Summary

Word embeddings have received a lot of attention since some Tomas Mikolov published [word2vec](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) in 2013  and showed that the embeddings that the neural network learned by "reading" a large corpus of text preserved semantic relations between words. As a result, this type of embedding started being studied in more detail and applied to more serious NLP and IR tasks such as summarization, query expansion, etc... More recently, researchers and practitioners alike have come to appreciate the power of this type of approach and have started a cottage industry of modifying Mikolov's original approach to many different areas.  

In this talk we will cover the implementation and mathematical details underlying tools like word2vec and some of the applications word embeddings have found in various areas. Starting from an intuitive overview of the main concepts and algorithms underlying the neural network architecture used in word2vec we will proceed to discussing the implementation details of the word2vec reference implementation in tensorflow.

Finally, we will provide a birds eye view of the emerging field of "<anything>2vec" (dna2vec, node2vec, etc...) methods that use variations of the word2vec neural network architecture.

# Program

1. Neural network architecture and algorithms underlying word2vec.
* Basic intuition
* Skip-gram
* Softmax
* Cross-Entropy
* BackProp
* Online sources for pre-trained embeddings

2. Properties and Applications of word embeddings 
* Visualization
* Analogies

3. Brief overview of tensorflow
* Installation
* Computational Graph
* Simple example (Linear fitting)

4. Detailed discussion of tensorflows [reference implementation](https://github.com/tensorflow/models/blob/master/tutorials/embedding/word2vec.py)

5. Overview of word2vec variations and their applications
* [Linguistic Change](https://www.perozzi.net/publications/15_www_linguistic.pdf)
* [node2vec](https://cs.stanford.edu/people/jure/pubs/node2vec-kdd16.pdf)
* [dna2vec](https://arxiv.org/abs/1701.06279)

# Slides

## O'Reilly AI New York City, 2018
<iframe src="//www.slideshare.net/slideshow/embed_code/key/AvHmgfyoGTwtkF" width="595" height="485" frameborder="0" marginwidth="0" marginheight="0" scrolling="no" style="border:1px solid #CCC; border-width:1px; margin-bottom:5px; max-width: 100%;" allowfullscreen> </iframe> 

## Previous Editions
* [O'Reilly AI San Francisco 2017](aisf17)
* [This Week in Machine Learning and AI](https://twimlai.com/twiml-talk-048-word2vec-friends-bruno-goncalves/)
* [Byte Academy 2017](byteacademy)
* [Ai With the Best 2017](aiwtb)

# Contact

Bruno Gonçalves [[web]](http://www.bgoncalves.com) [[email]](mailto:bgoncalves@gmail.com) [[twitter]](https://twitter.com/bgoncalves)