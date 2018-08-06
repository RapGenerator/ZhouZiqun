# DeeCamp 2018 Personal repo for Ziqun Zhou  
## Docs  
Documents and notes. Mainly notes written with OneNote.  
## Code  
Python code. ~~Mainly pytorch~~. Now mainly TensorFlow.  
## Diary  
### Before 2018-08-02  
- Data cleaning. Remove dirty sentences and add pinyin notation(by Junli Wang).  
- Walk through RNN, LSTM and GRU in great mathematical detail  
- Implement char-RNN, LSTM in tf and pytorch  
- Dive into the docs of pytorch  
### 2018-08-02  
- Moved from PyTorch to TensorFlow QAQ.  
- Group meeting with tutor Wang. Re-unite the group and focus on one daily target everyday now.  
- Start to learn seq2seq emplemented with tf by Yunsheng pwp.  
- Group Q&A meeting. Discuss questions 1-6(RNN, LSTM) but some remains unclear. Need to use tutor for more detailed answer.  
### 2018-08-03  
- Being assigned to algorithm group. Prepare to read paper about skip through vector.  
- Prepare to write final website for presentation using Vue.js as frontend framework.  
- Dive into Ruichang's seq2seq model in tf but not finished yet.  
- Discussed with Zili and got a very deep understanding about LSTM as well as how to implement LSTM from scratch.  
- Make weekly video by sing 《祖传程序员》 :)  
### 2018-08-04 - 2018-08-05  
- Make a install image for UCloud GPU VM including CUDA 9, cudnn 7 and tensorflow-gpu  
- Add teacher forcing to Yunsheng's model using ScheduledEmbeddingTrainingHelper  
- Try to train Yunsheng's model on both local GPU and UCloud GPU, however there seems to be some problems when restoring checkpoints in UCloud GPU. Need discussion tomorrow  
### 2018-08-06  
- Work with Yunsheng to finilize our baseline model (seq2seq with attention and beam search). Finally make it works on UCloud GPU. The bug is that dynamically creating dictionary is unreliable. Now we generate a static dictionary into a txt file and then load it from file to prevent disparity.  
- Meeting with out tutor. Solved some questions. And some new ideas are introduced by our tutor (e.g. RNN slot filling, skip-thoughts vector)  
- Finished reading the paper "Skip thoughts vector".  
