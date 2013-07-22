from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
 
setup(ext_modules=[Extension("fastaffine",
                             ["fastaffine.pyx"],
                             extra_compile_args='/openmp'],
                             language="c",),
                   ],
      cmdclass = {'build_ext': build_ext})
