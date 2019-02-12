from keras import layers, models, optimizers
from keras import backend as K

class Critic:
    """Critic (Value) Model."""

    def __init__(self, s_size, a_size):
        """Initialize parameters and build model.
        Params
        ======
            s_size (int): Dimension of each state
            a_size (int): Dimension of each action
        """
        self.s_size = s_size
        self.a_size = a_size

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.s_size,), name='states')
        actions = layers.Input(shape=(self.a_size,), name='actions')

        # Add hidden layer(s) for state pathway
        s_net = layers.Dense(units=256,kernel_regularizer=layers.regularizers.l2(1e-6))(states)
        s_net = layers.Activation("relu")(s_net)
        s_net = layers.Dropout(0.5)(s_net)

        s_net = layers.Dense(units=192, kernel_regularizer=layers.regularizers.l2(1e-6))(s_net)
        s_net = layers.Activation("relu")(s_net)

        # Add hidden layer(s) for action pathway
        a_net = layers.Dense(units=256,kernel_regularizer=layers.regularizers.l2(1e-6))(actions)
        a_net = layers.Activation("relu")(a_net)
        a_net = layers.Dropout(0.5)(a_net)
        
        a_net = layers.Dense(units=192,kernel_regularizer=layers.regularizers.l2(1e-6))(a_net)
        a_net = layers.Activation("relu")(a_net)
        
        # Combine state and action pathways
        net = layers.Add()([s_net, a_net])
        net = layers.Activation('relu')(net)

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values',
                                kernel_initializer=layers.initializers.RandomUniform(minval=-0.003, maxval=0.003)
                               )(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=.001)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        a_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=a_gradients)