# 让这个目录成为一个包，这样 setuptools 装得上。
# 里面的模块之间仍然用裸名互相 import —— 那靠 by1paths
# 把每个子目录放上 sys.path（装没装都一样）。
