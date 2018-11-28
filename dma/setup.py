
from setuptools import Extension, setup

module = Extension("dma",  
                  sources=[
                    'libdma/libdma.c',
                    'dma_module.c'
                  ])
setup(name='dma',  
     version='1.0',
     description='Python wrapper for custom C extension library to controlPL DMA.',
     ext_modules=[module])
