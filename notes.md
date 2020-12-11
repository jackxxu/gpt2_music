

## GPT2 model finetuning

### training library selection

There are various approaches to finetune GPT2 model. one way is to finetuning it during against the pretrained model from the [GPT2 repo](git@github.com:jackxxu/gpt2_music.git). However, it will require that some work in setup. We could also use the [Huggingface GPT2 transformers](https://huggingface.co/transformers/model_doc/gpt2.html) to do similar things. However, it similarly will probably require certain level of setup before starting to run. 

What we have found is this library called [`gpt-2-simple`](https://github.com/minimaxir/gpt-2-simple). The library is quite popular with 2.4k stars. We decided to try this library because it probably a nice wrapper around the GPT2 pretrained model, and provides 2 convenience methods `finetune` for finetuning, and `generate` to auto generate text. In addition, it also provides a convenient command line so that we don't have to write code.


### data preparation

in order to feed music information into GPT2 model, we first need to convert the data into a text format that GPT2 model is able to proceed. for that, we used [`music21` library from MIT](http://web.mit.edu/music21/). `music21` parses the mid file into notes and chords (concurrent notes) and other types, which we can use to feel into the model. 

There are a few types of the music elments that we are interested in: 
1. notes: a single note that is played at a particular moment for a duration. We represent it by using its note + its octave, such as "C#3", "D-2", or "E3". 
2. chords: a collection of the notes that are played concurrently. We represent it by simply concatenating the aforementioned note representations with space.
3. other notes are displayed with their `music21` name. 

here is an example sequence of notes that is converted from one for sample midi files.

```
"Piano, tempo.MetronomeMark animato Quarter=120.01, C major, meter.TimeSignature 4/4, rest, rest, rest, rest, rest, tempo.MetronomeMark larghetto Quarter=60.0, D4, rest, G3, A3, B3, B4, G4, C5, A3, A3 C4 E4, G4 E4, rest, F3 A3 C4, E4, D4, G4, B3, C5, D5, E5, F5, E5, G4, A4, G4, E4, G4, E4, C4, A3, G4, E4, D4, C4, G3 B3 D4, A3, A3 C4, G3, F3, A3, A3, A3"
```

We then concatenate all these elements with command and output them into a single-column CSV file. Then we are ready for feeding the songs into the model for finetuning.


### finetuning and generation

The finetune is actual quite straightforward, `gpt2.finetune` is all we need, after we download the [pretrained model from google cloud](https://storage.googleapis.com/gpt-2). The `finetune` method takes a file name and also the number of iteractions to train the model, we used 1000 iteractions. 

After finetuning is finished, the model is deposited into the checkpoint model. we can the called the `gpt2.generate` method to auto generate music notes. By default, it will start to generate music by itself. We can optionally specify a prefix, which sets the context for the music notes that is generate afterwards. 

It is interesting to note that the output notes is in the same semantic format as the input. This indicates that the finetuning model has learned the "grammar" of quite different domain. 


## Visualization

Visualziation helps humans to see the patterns in the model. We have found, [BertViz](https://github.com/jessevig/bertviz), an open-source tool for visualizing self-attention in the BERT language representation model. BertViz works by displaying Bert self-attention in a correlation graph, so that the contribution of each token to the prediction is shown.

BertViz extends earlier work by visualizing attention at three levels of granularity: the attention-head level, the model level, and the neuron level. 

Even though BertViz is designed for Bert as its name suggests, it works for any attention-based NLP neural network. In fact, BertViz supports HuggingFace GPT2 transformer with this [GPT2Model](https://github.com/jessevig/bertviz/blob/master/bertviz/transformers_neuron_view/modeling_gpt2.py) and [GPT2Tokenizer](https://github.com/jessevig/bertviz/blob/master/bertviz/transformers_neuron_view/tokenization_gpt2.py). 


However, gpt-2-simple package doesn't provide the attention output as huggingface does. It only outputs the generate text by default and attention information is not part of the output that can be requested.

Therefore, we had to do some modification. Without getting into too much details, gpt-2-simple code provides 2 layers in the outcome of the computation, which contains the past and present for each layer. The past being the input to the layer and output being the output to the layer. In a way, this is the attention matrix for GPT-2 model. With some transposition (the main code is below), we are able to get the attention matrix with number of tokens * the number of tokens as its dimensions. 

```python
if return_attention:
    # past_n_present should be in the dimension of
    # [batch, layers, 2, heads, sequence, features]
    past = past_n_present[:, :, :1, :, :, :]
    present = past_n_present[:, :, 1:, :, :, :]
    # compute the past and present attetntion
    attention = tf.matmul(past/tf.cast(temperature, tf.float32), tf.transpose(present, perm=[0, 1, 2, 3, 5, 4]))
    attention = tf.nn.softmax(attention, axis=-1)
    return tokens, attention
else:
    return tokens
```

with that attention matrix, Bertviz readily outputs the attention graph.

<img src="images/layer0.png" alt="layer0" width="200"/>

## Discussion

The attention in music seems to be a bit more scattered than language, where there doesn't seem to be a pattern for the notes to pay close attention to. However, one observation is that the later the layers, the more concentrated the attention becomes. see below of the comparison of 3 layers. 

<table>
	<tr>
		<th>
			Layer 0
		</th>
		<th>
			Layer 6
		</th>
		<th>
			Layer 11
		</th>
	</tr>
	<tr>
		<td>
			<img src="images/layer0.png" alt="layer0" height="400"/>
		</td>
		<td>
			<img src="images/layer6.png" alt="layer6" height ="400"/>
		</td>
		<td>
			<img src="images/layer11.png" alt="layer11" height ="400"/>
		</td>
	</tr>
</table>


This can be explained by that the later layers have learned the more corelations between the notes, and therefore the attentions are more distinct.

## Future Improvements

For future improvements, I'd like to work on the following:

1. encode the note itself: with the default tokenization, GPT2 doesn't recognizes music notes. for example: `C#2` is actually tokenized as 3 separate notes: `C`, `#`, and `3`, through the byte-pairing encoding. This is not ideal, as these 3 tokens are interpreted by the default language embedding, which doesn't apply to the music context. It will take a lot of training data to provide the context. It probably would be ideal to tokenize a music note such as `C#2` as one note. We can do that with the GPT2 `encoder.json` and `vocab.bpe` file. It will be interesting to see how GPT2 performs with distinct music notes encoding, with something like this

```python
note_to_token = dict()
token_to_note = dict()

token = 50256

for octave in range(8):
  for main in 'ABCDEFG':
    for minor in ['#', '', '-']:
      token += 1
      note = f'{main}{minor}{octave}'
      note_to_token[note] = token
      token_to_note[token] = note

print(note_to_token)
print(token_to_note)

## =>
{'A#0': 50257, 'A0': 50258, 'A-0': 50259, 'B#0': 50260, 'B0': 50261, 'B-0': 50262, ...}
{50257: 'A#0', 50258: 'A0', 50259: 'A-0', 50260: 'B#0', 50261: 'B0', 50262: 'B-0', ...}

``` 

2. 

 