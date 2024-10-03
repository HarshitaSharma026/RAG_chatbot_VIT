# This file list down all the problems i faced during making this chatbot, and then solutions are provided for each problem.

## Overlapping of subjects within courses
### Solution 1:
Adding name of the course in multiple places in the text file (especially in the with the subjects), to make the model understand which course does this subject syllabus belongs to.
### Solution 2:
Using PromptTemplate, providing system and general prompt to the model greatly reduced this error. Two types of prompts provided:
1. Where the model is instructed to recontruct the sentence into a standalone sentence if any question is asked based on the previous answer / question.
2. System prompt to instruct the model exactly how to answer the user question.

## Getting incomplete answers
solutions:
1. Chunk size - less chunk size, more accurate - 512, helps improve efficiency and accuracy 
2. Documents didn't properly break down into chunks