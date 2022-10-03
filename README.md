# Audio Collager

![](https://github.com/jbordoe/audio-collage/blob/master/docs/collager_b.png?raw=true)

Attempt to reconstruct target audio using snippets of source audio.

## Setup

Install Python >3.9 and the dependencies with [poetry](https://python-poetry.org/):

```python
poetry install
```

## Run
`puython collager.py COMMAND`

Run `python collager.py collage` for more details.

### Examples
#### Creating a collage
```bash
python collager.py collage -t target.wav -s source.wav -o collage.wav
```

#### Chopping audio
Chop the given file in to snippets of 250 milliseconds
```bash
python collager.py chop -l 250 -f sample.wav -o sample_slices/
```
