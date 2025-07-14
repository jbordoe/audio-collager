# Audio Collager

![](https://github.com/jbordoe/audio-collage/blob/master/docs/collager_b.png?raw=true)

Attempt to reconstruct target audio using snippets of source audio.

## Background

This tool is inspired by the Scott Johnson album *[John Somebody](https://scottjohnsoncomposer.com/compositions/johnsomebody.html)* ([this track](https://scottjohnsoncomposer.com/compositions/audioclips/InvoluntarySong.ogg) in particular) in which samples of speech and laughter are cut up, rearranged and looped to create rhythm and melody.


## Setup

Install Python >3.9 and the dependencies with [poetry](https://python-poetry.org/):

```python
poetry install
```

## Run
`poetry run audio-collage COMMAND [OPTIONS]`

Run `poetry run audio-collage --help` for more details.

### Examples
#### Creating a collage
```bash
poetry run audio-collage collage -t target.wav -s source.wav -o collage.wav
```

#### Chopping audio
Chop the given file in to snippets of 250 milliseconds
```bash
poetry run audio-collage chop -l 250 -f sample.wav -o sample_slices/
```

### Use Cases

Let's begin with two breakbeats:

* Black Heat [Zimba Ku](docs/audio/breaks/black_heat__zimba_ku.wav) ([source](https://www.youtube.com/watch?v=mybkf-H8mkA))
* The Winstons [Amen Brother](docs/audio/breaks/amen_brother.wav) ([source](https://www.youtube.com/watch?v=GxZuq57_bYM))

We want to recreate the drum pattern of the former using audio from the latter.

```bash
poetry run audio-collage collage -t docs/audio/breaks/amen_brother.wav -s docs/audio/breaks/black_heat__zimba_ku.wav -o docs/audio/breaks/out/amen_zimba.wav
```
You can listen to the output [here](docs/audio/breaks/out/amen_zimba.wav)

You can quickly run this example locally using:

```bash
poetry run audio-collage example
```
