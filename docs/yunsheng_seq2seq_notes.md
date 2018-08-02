## Questions  
### Q1. Why making Seq2SeqModel a subclass of "object"?  
It makes the class to be new-style class, but it's unnecessary after Python 3.2. See <https://stackoverflow.com/questions/2588628/what-is-the-purpose-of-subclassing-the-class-object-in-python> for more detail.  
### Q2. Why using 'None' in argument 'shape' of tf.placeholder? Such as tf.placeholder(tf.int32, [None, None])  
It makes it possible for not being bounded to a fixed-size batches and reusing the learnt network to predict either single instances or batch of instances. See <https://stackoverflow.com/questions/44461197/why-use-none-for-the-batch-dimension-in-tensorflow> for more detail. 
### Q3. model.py line 84: Why the size of encoder_outputs is batch_size*encoder_inputs_length*rnn_size? Does this means that encoder_inputs_length = encoder_outputs_length?  
### Q4. See Notes.1 Why error occurs when directly passing a LSTM cell variable in MultiRNNCell? Is it because that all cells in MultiRNNCell will actually share one single var 'cell'?  

## Notes
- 关于RNN cell的创建: 创建单个cell，这里需要注意的是一定要使用一个single_rnn_cell的函数，不然直接把cell放在MultiRNNCell的列表中最终模型会发生错误（亲测确实有此坑）  

