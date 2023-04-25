import torch.nn as nn


class FFN(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        ffn_hidden_size=100,
        num_layers=3,
        dropout=0.1,
        activation="ReLU",
        bias=False,
    ):
        super(FFN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = ffn_hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = nn.Dropout(p=dropout)
        self.activation = self.get_activation_function(activation)

        if self.num_layers == 1:
            ffn = [
                self.dropout,
                nn.Linear(self.input_size, self.output_size, bias=self.bias),
            ]
        else:
            ffn = [
                self.dropout,
                nn.Linear(self.input_size, self.hidden_size, bias=self.bias),
            ]
            for _ in range(self.num_layers - 2):
                ffn.extend(
                    [
                        self.activation,
                        self.dropout,
                        nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias),
                    ]
                )
            ffn.extend([self.activation, self.dropout])
        self.ffn = nn.Sequential(*ffn)
        self.output_layer = nn.Linear(
            self.hidden_size, self.output_size, bias=self.bias
        )

    def forward(self, inp):
        output = self.ffn(inp)
        output = self.output_layer(output)
        return output

    def get_activation_function(self, activation) -> nn.Module:
        """
        Gets an activation function module given the name of the activation.

        :param activation: The name of the activation function.
        :return: The activation function module.
        """
        if activation == "ReLU":
            return nn.ReLU()
        elif activation == "LeakyReLU":
            return nn.LeakyReLU(0.1)
        elif activation == "PReLU":
            return nn.PReLU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "SELU":
            return nn.SELU()
        elif activation == "ELU":
            return nn.ELU()
        else:
            raise ValueError(f'Activation "{activation}" not supported.')
