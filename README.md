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
### 2018-08-07
- Generate two samples (<http://pw.qwqq.pw/pavrjzcuf> and <http://pw.qwqq.pw/psbwtra8k>) with seq2seq model (attention, LSTM, Beam Search, rnn_size=256, num_layers=4, final_loss=3.1)  
- Finished reading paper "Topic to essay generation with neural network". Ready to go through the relating tf code.  
- Shijun & Weiwen shared their ideas towards paper "Chinese Poetry Generation with Planning based NN". Ready to write code about keywords extraction.  
- Mengxin shared her ideas towards paper "An interactive Poetry Generation System".  Ready to write code about modified beam search with score mentioned in that paper.  
- Planning to change the encoder to Bi-LSTM in the previous model.  
- Planning to study Yunsheng's skip-thoughts model implementation.  
### 2018-08-08  
- Implement Bi-LSTM Encoder and trained for a about 80 epochs. Some improvements towards the fluency of the sentences have emerged.  
- Study Yunsheng's Skip Thought model. Great improvements towards the semantic association between sentences have emerged.  
- Combine two models mentioned above to generate Bi-SkipThought model. Training in progress.  
- Discuss about keyword generation with Yunsheng and Weiwen. Planning to implement a keyword-to-sentence model tomorrow. Need further discussion with tutor.   
### 2018-08-09  
- Discuss with tutor. Three topic oriented generation model (Baidu's, HIU's and tutor's) are proposed. Planning to try the third model.  
- Start to write frontend with Vue.js.  
- Try to train Bi-SkipThought but the loss is still too high QAQ.   
### 2018-08-10 - 2018-08-12  
- Modified Bi-RNN_Encoder, canceling reversed input and reducing number of layers to 2. The [result](http://pw.qwqq.pw/pkwyti2mj) seems prety exciting.  
- Tried to apply the same trick to Bi-SkipThought but the results hardly improved and the loss is still high compared to not using SkipThought.  
- Trying to understand Custom BeamSearch code given by tutor. There are some questions needing furthur discuss.  
### 2018-08-13  
- Continue to develop frontend. Here's the [github repo](https://github.com/zhouziqunzzq/rg-frontend).  
- Keyword selector & Template selector are nearly finished. Working on Lyrics player.  
