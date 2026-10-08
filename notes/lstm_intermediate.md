\# Long Short-Term Memory Networks



\## Overview

Long Short-Term Memory, or LSTM, is a recurrent neural network

architecture designed to model sequential data. It maintains a

cell state that helps carry information across time steps.



\## Forget Gate

The forget gate determines which information from the previous

cell state should be retained or discarded. It uses a sigmoid

activation to produce values between zero and one. Values close

to zero discard information, while values close to one retain it.



\## Input Gate

The input gate controls how much new information is written into

the cell state. Candidate information is combined with the input

gate values to update the cell state.



\## Output Gate

The output gate controls which information from the updated cell

state contributes to the current hidden state.



\## Applications

LSTMs can be used for time-series forecasting, language modeling,

and other tasks involving sequential data.

