## Questions  
### Q1. Why making Seq2SeqModel a subclass of "object"?  
It makes the class to be new-style class, but it's unnecessary after Python 3.2. See <https://stackoverflow.com/questions/2588628/what-is-the-purpose-of-subclassing-the-class-object-in-python> for more detail.  
### Q2. Why using 'None' in argument 'shape' of tf.placeholder? Such as tf.placeholder(tf.int32, [None, None])  
It makes it possible for not being bounded to a fixed-size batches and reusing the learnt network to predict either single instances or batch of instances. See <https://stackoverflow.com/questions/44461197/why-use-none-for-the-batch-dimension-in-tensorflow> for more detail. 
### Q3. model.py line 84: Why the size of encoder_outputs is batch_size*encoder_inputs_length*rnn_size? Does this means that encoder_inputs_length = encoder_outputs_length?  
### Q4. See Notes.1 Why error occurs when directly passing a LSTM cell variable in MultiRNNCell? Is it because that all cells in MultiRNNCell will actually share one single var 'cell'?  
### Q5. What does param 'attention_layer_size' for AttentionWrapper mean?  
### Q6. model.py line 176: Why need tf.expand_dims on decoder_outputs.sample_id?  
### Q7. data_helper.py line 52: Can be simplified with word_dic_new = [k for k, v in word_dic.items() if v > 1]  

## Notes
- 关于 RNN cell 的创建: 创建单个 cell，这里需要注意的是一定要使用一个single_rnn_cell的函数，不然直接把cell放在MultiRNNCell的列表中最终模型会发生错误（亲测确实有此坑）  
- 关于 Attention Mechanism 和 Attention Wrapper 的详细讲解：<https://blog.csdn.net/qsczse943062710/article/details/79539005>  
- 关于 tf.identity 的作用：<https://blog.csdn.net/wuguangbin1230/article/details/79793737>
- 关于 tf.Tensor 和 tf.Variable 的详细区别: <https://stackoverflow.com/questions/40866675/implementation-difference-between-tensorflow-variable-and-tensorflow-tensor>
- 在 train 的时候 beam_search 应该为 False ，否则会出现维度不一致！  
