### To create the glossary for the RAG, we need : 
-Database creation : To get some files, documents, ... for the glossary  
-Chunking : Make chunking (decouper les fichiers)  
-Vectorization : embed the chunks  
-Index : Save this in an "Index"  

#### Database creation : 
It is proposed to take the labelled training examples from HIC and IShate (take only training data for no bias)
Hate speech lexicons (HatBase, HurtLex, Davidson lexicon)
Definitions and typology

#### Chunking :
Because some sentences can be too long to be encoded in a vector, we sometimes need to chunk them.
The goal is to have a dataset which has easy chunking (typical size of a chunck is 100-512 tokens) to facilitate chunking
When the chunk is not easy to do, we have to think what to do

#### Vectorization 
We just need to pass the chunks into the model and to get the output and store it into a FAISS file. It is automatically vectorized by the model's vector space so it has "sense" for him

#### FAISS