import sys

if sys.argv[1:2] == ["play"]:
    del sys.argv[1]
    from .play import main
else:
    from .uci import main

main()
