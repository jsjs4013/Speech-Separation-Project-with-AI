import tensorflow as tf

class VainillaT5(tf.keras.Model):
    def train_step(self, data):
        """print('inp',inp.shape) 
        startMask = tf.cast(tf.fill([1,258],-1),dtype=tf.float32)
        endMask = tf.cast(tf.fill([1,258],-2),dtype=tf.float32)
        tar_inp = tf.concat([startMask, tar],0)
        tar_real = tf.concat([tar, endMask],0)
        print('tar_inp',tar_inp.shape)
        print('tar_real',tar_real[0])"""
        inp, tar, length = data
        startMask = tf.cast(tf.fill([tf.shape(tar)[0], 1, tf.shape(tar)[-1]],-1),dtype=tf.float32)
        tar = tf.concat([startMask, tar],1)

        tar_inp = tar[:, :-2, :]
        s1_target = tf.slice(tar_inp, [0, 0, 0], [-1, -1, 129])
        s2_target = tf.slice(tar_inp, [0, 0, 129], [-1, -1, -1])
        tar_inp2 = tf.concat([s2_target, s1_target], -1)
        tar_real = tar[:, 1:, :]

        """
        enc_padding_mask, combined_mask, dec_padding_mask = None, None, None
        combined_mask = create_look_ahead_mask(tf.shape(tar_inp)[1])
        combined_mask = combined_mask[tf.newaxis, tf.newaxis, :, :]
        combined_mask = tf.repeat(combined_mask, tf.shape(tar_inp)[0], 0)
        """
        with tf.GradientTape() as tape:
            predictions1 = self((inp, tar_inp, length), training=True)
            predictions2 = self((inp, tar_inp2, length), training=True)
            real_predict = tf.concat([predictions1, predictions2], 0)
            
            loss = self.compiled_loss(tar_real, real_predict, regularization_losses=self.losses)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(tar_real, real_predict)

        return {m.name: m.result() for m in self.metrics}
        #train_accuracy(accuracy_function(tar_real, predictions))


    def test_step(self, data):
        inp, tar, length = data
        startMask = tf.cast(tf.fill([tf.shape(tar)[0], 1, tf.shape(tar)[-1]],-1),dtype=tf.float32)
        tar = tf.concat([startMask, tar],1)

        tar_inp = tar[:, :-2, :]
        s1_target = tf.slice(tar_inp, [0, 0, 0], [-1, -1, 129])
        s2_target = tf.slice(tar_inp, [0, 0, 129], [-1, -1, -1])
        tar_inp2 = tf.concat([s2_target, s1_target], -1)
        tar_real = tar[:, 1:, :]

        """
        encoder_input = inp
        max_length = tf.shape(tar_inp)[1]
        
        # as the target is english, the first word to the transformer should be the
        # english start token.
        # start here
        i = 0
        output = startMask
        zero_clipping = tf.constant([0.])
        while i < max_length :
            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions = self((encoder_input, output, length), training=False)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
            predictions = tf.math.maximum(predictions, zero_clipping)
            predicted_id = predictions

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=1)
            i = i + 1
        """

        predictions1 = self((inp, tar_inp, length), training=False)
        predictions2 = self((inp, tar_inp2, length), training=False)
        real_predict = tf.concat([predictions1, predictions2], 0)

        # Updates stateful loss metrics.
        self.compiled_loss(tar_real, real_predict, regularization_losses=self.losses)

        self.compiled_metrics.update_state(tar_real, real_predict)
        # Collect metrics to return
        return {m.name: m.result() for m in self.metrics}

        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics
    
    def predict_step(self, data):
        inp, tar, length = data
        startMask = tf.cast(tf.fill([tf.shape(tar)[0], 1, tf.shape(tar)[-1]],-1),dtype=tf.float32)
        tar = tf.concat([startMask, tar],1)

        tar_inp = tar[:, :-1, :]

        return self((inp, tar_inp, length), training=False)


class T5ChangedSTFT(tf.keras.Model):
    def train_step(self, data):
        """print('inp',inp.shape) 
        startMask = tf.cast(tf.fill([1,258],-1),dtype=tf.float32)
        endMask = tf.cast(tf.fill([1,258],-2),dtype=tf.float32)
        tar_inp = tf.concat([startMask, tar],0)
        tar_real = tf.concat([tar, endMask],0)
        """
        inp, tar, length = data
        startMask = tf.cast(tf.fill([tf.shape(tar)[0], 1, tf.shape(tar)[-1]],-1),dtype=tf.float32)
        tar = tf.concat([startMask, tar],1)

        tar_inp = tar[:, :-2, :]
        tar_real = tar[:, 1:, :]
        

        """
        enc_padding_mask, combined_mask, dec_padding_mask = None, None, None
        combined_mask = create_look_ahead_mask(tf.shape(tar_inp)[1])
        combined_mask = combined_mask[tf.newaxis, tf.newaxis, :, :]
        combined_mask = tf.repeat(combined_mask, tf.shape(tar_inp)[0], 0)
        """
        with tf.GradientTape() as tape:
            prediction = self((inp, tar_inp, length), training=True)
            
            loss = self.compiled_loss(tar_real, prediction, regularization_losses=self.losses)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(tar_real, prediction)

        return {m.name: m.result() for m in self.metrics}
        #train_accuracy(accuracy_function(tar_real, predictions))


    def test_step(self, data):
        inp, tar, length = data
        startMask = tf.cast(tf.fill([tf.shape(tar)[0], 1, tf.shape(tar)[-1]],-1),dtype=tf.float32)
        tar = tf.concat([startMask, tar],1)

        tar_inp = tar[:, :-2, :]
        tar_real = tar[:, 1:, :]
        
        """tar_inp = tar[:, 1:-1, :]
        tar_real = tar[:, :, :]
        
        tar_inp = tf.concat([startMask, tar_inp],1)"""

        """
        encoder_input = inp
        max_length = tf.shape(tar_inp)[1]
        
        # as the target is english, the first word to the transformer should be the
        # english start token.
        # start here
        i = 0
        output = startMask
        zero_clipping = tf.constant([0.])
        while i < max_length :
            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions = self((encoder_input, output, length), training=False)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
            predictions = tf.math.maximum(predictions, zero_clipping)
            predicted_id = predictions

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=1)
            i = i + 1
        """

        predictions = self((inp, tar_inp, length), training=False)

        # Updates stateful loss metrics.
        self.compiled_loss(tar_real, predictions, regularization_losses=self.losses)

        self.compiled_metrics.update_state(tar_real, predictions)
        # Collect metrics to return
        return {m.name: m.result() for m in self.metrics}

        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics
    
    def predict_step(self, data):
        inp, tar, length = data
        startMask = tf.cast(tf.fill([tf.shape(tar)[0], 1, tf.shape(tar)[-1]],-1),dtype=tf.float32)
        tar = tf.concat([startMask, tar],1)

        tar_inp = tar[:, :-1, :]

        return self((inp, tar_inp, length), training=False)


class T5YesMaskSTFT(tf.keras.Model):
    def train_step(self, data):
        """print('inp',inp.shape) 
        startMask = tf.cast(tf.fill([1,258],-1),dtype=tf.float32)
        endMask = tf.cast(tf.fill([1,258],-2),dtype=tf.float32)
        tar_inp = tf.concat([startMask, tar],0)
        tar_real = tf.concat([tar, endMask],0)
        """
        inp, tar, length = data
        startMask = tf.cast(tf.fill([tf.shape(tar)[0], 1, tf.shape(tar)[-1]],-1),dtype=tf.float32)
        tar = tf.concat([startMask, tar],1)

        tar_inp = tar[:, :-2, :]
        tar_real = tar[:, 1:, :]
        

        """
        enc_padding_mask, combined_mask, dec_padding_mask = None, None, None
        combined_mask = create_look_ahead_mask(tf.shape(tar_inp)[1])
        combined_mask = combined_mask[tf.newaxis, tf.newaxis, :, :]
        combined_mask = tf.repeat(combined_mask, tf.shape(tar_inp)[0], 0)
        """
        with tf.GradientTape() as tape:
            prediction = self((inp, tar_inp, length), training=True)
            
            loss = self.compiled_loss(tar_real, prediction, regularization_losses=self.losses)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(tar_real, prediction)

        return {m.name: m.result() for m in self.metrics}
        #train_accuracy(accuracy_function(tar_real, predictions))


    def test_step(self, data):
        inp, tar, length = data
        startMask = tf.cast(tf.fill([tf.shape(tar)[0], 1, tf.shape(tar)[-1]],-1),dtype=tf.float32)
        tar = tf.concat([startMask, tar],1)

        tar_inp = tar[:, :-2, :]
        tar_real = tar[:, 1:, :]
        
        """tar_inp = tar[:, 1:-1, :]
        tar_real = tar[:, :, :]
        
        tar_inp = tf.concat([startMask, tar_inp],1)"""

        """
        encoder_input = inp
        max_length = tf.shape(tar_inp)[1]
        
        # as the target is english, the first word to the transformer should be the
        # english start token.
        # start here
        i = 0
        output = startMask
        zero_clipping = tf.constant([0.])
        while i < max_length :
            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions = self((encoder_input, output, length), training=False)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
            predictions = tf.math.maximum(predictions, zero_clipping)
            predicted_id = predictions

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=1)
            i = i + 1
        """

        predictions = self((inp, tar_inp, length), training=False)

        # Updates stateful loss metrics.
        self.compiled_loss(tar_real, predictions, regularization_losses=self.losses)

        self.compiled_metrics.update_state(tar_real, predictions)
        # Collect metrics to return
        return {m.name: m.result() for m in self.metrics}

        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics
    
    def predict_step(self, data):
        inp, tar, length = data
        startMask = tf.cast(tf.fill([tf.shape(tar)[0], 1, tf.shape(tar)[-1]],-1),dtype=tf.float32)
        tar = tf.concat([startMask, tar],1)

        tar_inp = tar[:, :-1, :]

        return self((inp, tar_inp, length), training=False)

class SourceBySourceT5(tf.keras.Model):
    def train_step(self, data):
        """print('inp',inp.shape) 
        startMask = tf.cast(tf.fill([1,258],-1),dtype=tf.float32)
        endMask = tf.cast(tf.fill([1,258],-2),dtype=tf.float32)
        tar_inp = tf.concat([startMask, tar],0)
        tar_real = tf.concat([tar, endMask],0)
        print('tar_inp',tar_inp.shape)
        print('tar_real',tar_real[0])"""
        inp, tar, length = data
        startMask = tf.cast(tf.fill([tf.shape(tar)[0], 1, tf.shape(tar)[-1]],-1),dtype=tf.float32)
        tar = tf.concat([startMask, tar],1)

        tar_inp = tar[:, :-2, :]
        s1_target = tf.slice(tar_inp, [0, 0, 0], [-1, -1, 129])
        s2_target = tf.slice(tar_inp, [0, 0, 129], [-1, -1, -1])
        tar_inp2 = tf.concat([s2_target, s1_target], -1)
        tar_real = tar[:, 1:, :]

        """
        enc_padding_mask, combined_mask, dec_padding_mask = None, None, None
        combined_mask = create_look_ahead_mask(tf.shape(tar_inp)[1])
        combined_mask = combined_mask[tf.newaxis, tf.newaxis, :, :]
        combined_mask = tf.repeat(combined_mask, tf.shape(tar_inp)[0], 0)
        """
        with tf.GradientTape() as tape:
            predictions1 = self((inp, tar_inp, length), training=True)
            predictions2 = self((inp, tar_inp2, length), training=True)
            real_predict = tf.concat([predictions1, predictions2], 0)
            
            loss = self.compiled_loss(tar_real, real_predict, regularization_losses=self.losses)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(tar_real, real_predict)

        return {m.name: m.result() for m in self.metrics}
        #train_accuracy(accuracy_function(tar_real, predictions))


    def test_step(self, data):
        inp, tar, length = data
        startMask = tf.cast(tf.fill([tf.shape(tar)[0], 1, tf.shape(tar)[-1]],-1),dtype=tf.float32)
        tar = tf.concat([startMask, tar],1)

        tar_inp = tar[:, :-2, :]
        s1_target = tf.slice(tar_inp, [0, 0, 0], [-1, -1, 129])
        s2_target = tf.slice(tar_inp, [0, 0, 129], [-1, -1, -1])
        tar_inp2 = tf.concat([s2_target, s1_target], -1)
        tar_real = tar[:, 1:, :]

        """
        encoder_input = inp
        max_length = tf.shape(tar_inp)[1]
        
        # as the target is english, the first word to the transformer should be the
        # english start token.
        # start here
        i = 0
        output = startMask
        zero_clipping = tf.constant([0.])
        while i < max_length :
            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions = self((encoder_input, output, length), training=False)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
            predictions = tf.math.maximum(predictions, zero_clipping)
            predicted_id = predictions

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=1)
            i = i + 1
        """

        predictions1 = self((inp, tar_inp, length), training=False)
        predictions2 = self((inp, tar_inp2, length), training=False)
        real_predict = tf.concat([predictions1, predictions2], 0)

        # Updates stateful loss metrics.
        self.compiled_loss(tar_real, real_predict, regularization_losses=self.losses)

        self.compiled_metrics.update_state(tar_real, real_predict)
        # Collect metrics to return
        return {m.name: m.result() for m in self.metrics}

        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics
    
    def predict_step(self, data):
        inp, tar, length = data
        startMask = tf.cast(tf.fill([tf.shape(tar)[0], 1, tf.shape(tar)[-1]],-1),dtype=tf.float32)
        tar = tf.concat([startMask, tar],1)

        tar_inp = tar[:, :-1, :]

        return self((inp, tar_inp, length), training=False)