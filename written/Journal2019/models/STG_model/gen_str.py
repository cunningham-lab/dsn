import sys

c = int(sys.argv[1])
rs = int(sys.argv[2])

for k in range(10):
    print("\includegraphics[scale=0.125]{DSN_figs/STGCircuit_DSN_c=%d_rs=%d_k=%d.png}" % (c, rs, k+1))
