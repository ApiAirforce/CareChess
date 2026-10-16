# CareChess

A simple chess engine we built to play with.

It's trained to play moderate, human-like moves, so you get a real game instead of getting crushed.

On [lichess.org](https://lichess.org/@/CareChess-Bot/playing) we play with the currently strongest version, beyond the opensource one.


## Getting started

```
pip install -r requirements.txt
python play.py
```

Type your moves like `e4`, `Nf3`, `O-O` or `e2e4`. `undo` takes back your last move (not with a clock), `quit` ends the game.

## Options

| Option | What it does |
|---|---|
| `--black` | you play Black |
| `--time 10` | 10 minutes each (`--time 1` for bullet) |
| `--inc 2` | 2 seconds added after every move |
| `--steps 2000` | how much it thinks per move when there's no clock |
| `--ascii` | letters instead of chess symbols |

## Playing in a chess GUI

CareChess speaks UCI, so you can add it to any chess program like Arena, Cute Chess or BanksiaGUI. Use this as the engine command:

```
python uci.py
```

It uses the game clock from the GUI. If you'd rather give it a fixed amount of thinking, set the `Steps` option.

## How it works

CareChess has two parts that work together:

- **A neural network** that looked at millions of positions from real games and learned which moves make sense. It decides which moves are worth considering.
- **A small search** that tries those moves out a few steps ahead and counts material, so it doesn't just hang its pieces.

The network is small (6.7M parameters), so it runs fine on a normal CPU, no graphics card needed.

## License

MIT, see [LICENSE](LICENSE). Built with [PyTorch](https://pytorch.org) and [python-chess](https://github.com/niklasf/python-chess)
