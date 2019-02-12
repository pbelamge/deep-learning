from keras import layers, models, optimizers
from keras import backend as K

class Actor:
    """Actor (Policy) Model."""
    
    # action size, state size, action low and high
    def __init__(self, s_size, a_size, a_low, a_high):
        """Initialize parameters and build model.
        Params
        ======
            s_size (int): Dimension of each state
            a_size (int): Dimension of each action
            a_low (array): Min value of each action dimension
            a_high (array): Max value of each action dimension
        """
        self.s_size = s_size
        self.a_size = a_size
        self.a_low = a_low
        self.a_high = a_high
        
        # action range
        self.a_range = self.a_high - self.a_low

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.s_size,), name='states')

        # Add hidden layers
        net = layers.Dense(units=256, kernel_regularizer=layers.regularizers.l2(1e-6))(states)
        net = layers.Activation("relu")(net)
        net = layers.Dropout(0.5)(net)

        net = layers.Dense(units=192,kernel_regularizer=layers.regularizers.l2(1e-6))(net)
        net = layers.Activation("relu")(net)

        # Add final output layer with sigmoid activation
        raw_actions = layers.Dense(units=self.a_size, activation='sigmoid',
                                   name='raw_actions',
                                   kernel_initializer=layers.initializers.RandomUniform(minval=-0.003, maxval=0.003)
                                  )(net)

        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.a_range) + self.a_low,
            name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        a_gradients = layers.Input(shape=(self.a_size,))
        loss = K.mean(-a_gradients * actions)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=.0001)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, a_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)