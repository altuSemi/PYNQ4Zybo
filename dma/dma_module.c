////////////////////////////////////////
//dma_lib: 	C Extension for dma access from python
//File name: 	dma_module.c
//Author: 	altuSemi
//Date:   	Nov 18,2018
//
////////////////////////////////////////                                        
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <sys/mman.h>
#include <string.h>
#include <stdint.h>
#include "libdma/libdma.h"
#include <numpy/arrayobject.h>




DMACtrlBuf DMAControlBuffer={
			0,		//Control Descriptor
			(uint32_t) 0,	//Control Pointer
 			0x04040000,	//Control Physical Address
			0,		//Buf select 0/1
			0,		//Buf length
			0,		//Buffer0 Descriptor
			0,		//Buffer1 Descriptor
			(uint32_t) 0,	//Buffer0 Pointer
			(uint32_t) 0, 	//Buffer1 Pointer
			0x01000000,	//Buffer0 Physical Address
			0x02000000	//Buffer1 Physical Address
};

static PyObject * DMALib_DMABufGet(PyObject *self, PyObject *args) {

  uint32_t * DstPointer=DMAControlBuffer.BufMMAPPointer[!DMAControlBuffer.BufSelect];
  //int *array = NULL;
  int i;

  //if (!(array = malloc(DMAControlBuffer.BufLength * sizeof(int)))) return NULL;
  //for (i = 0; i < DMAControlBuffer.BufLength; ++i) array[i] = DstPointer[i];
  // return the array as a numpy array (numpy will free it later)
  npy_intp dims[1] = {DMAControlBuffer.BufLength};
  PyObject *narray = PyArray_SimpleNewFromData(1, &dims[0], NPY_INT, (int*)DstPointer);
  // this is the critical line - tell numpy it has to free the data
  PyArray_ENABLEFLAGS((PyArrayObject*)narray, NPY_ARRAY_OWNDATA);
  return narray;

}

static PyObject *DMALib_DMABufSwap(PyObject *self, PyObject *args) {
    DMAControlBuffer.BufSelect=!DMAControlBuffer.BufSelect;
    return Py_BuildValue("i", true);
}

static PyObject *DMALib_DMABufSet(PyObject *self, PyObject *args) {
    PyObject *lst;
    if (!PyArg_ParseTuple(args, "O", &lst)) {
        return NULL;
    }

    int n = PyObject_Length(lst);
    if (n < 0) {
        return NULL;
    }
    bool SrcBuf=0;
    uint32_t * SourcePointer=DMAControlBuffer.BufMMAPPointer[DMAControlBuffer.BufSelect];
    for (int i = 0; i < n; i++) {
        PyLongObject *item = PyList_GetItem(lst, i);
        long num = PyLong_AsLong(item);
        SourcePointer[i]=num;
    }
    DMAControlBuffer.BufLength=n;
    return Py_BuildValue("i", true);
}

static PyObject *DMALib_DMATransfer(PyObject *self, PyObject *args) {  
    DMATransfer();
    return Py_BuildValue("i", true);
    //return PyLong_FromVoidPtr(nums_out);
}

static PyMethodDef DMALib_FunctionsTable[] = {  
    {
        "DMABufSwap",      // name exposed to Python
        DMALib_DMABufSwap, // C wrapper function
        METH_VARARGS,          // received variable args (but really just 1)
        "Swap source buffers 0/1" // documentation
    },{
        "DMABufGet",      // name exposed to Python
        DMALib_DMABufGet, // C wrapper function
        METH_VARARGS,          // received variable args (but really just 1)
        "Get DMA buffer content from destination buffer" // documentation
    },{
        "DMABufSet",      // name exposed to Python
        DMALib_DMABufSet, // C wrapper function
        METH_VARARGS,          // received variable args (but really just 1)
        "Set DMA source buffer" // documentation
    },{
        "DMATransfer",      // name exposed to Python
        DMALib_DMATransfer, // C wrapper function
        METH_VARARGS,          // received variable args (but really just 1)
        "Transfer a list from Python to the PL, get a list back from the PL" // documentation
    },{
        NULL, NULL, 0, NULL
    }
};

static struct PyModuleDef DMALib_Module = {  
    PyModuleDef_HEAD_INIT,
    "dma",     // name of module exposed to Python
    "Python wrapper for custom C extension library to controlPL DMA.", // module documentation
    -1,
    DMALib_FunctionsTable
};

PyMODINIT_FUNC PyInit_dma(void) {  
    //run init functions:
    //numpy array init:
    import_array();
    //DMA control and buffers struct init:
    DMAControlBuffer.CtrlDescriptor = open("/dev/uio0", O_RDWR | O_SYNC); // Open /dev/uio0 which represents the dma control area
    DMAControlBuffer.CtrlMMAPPointer = mmap(NULL, 65535, PROT_READ | PROT_WRITE, MAP_SHARED,DMAControlBuffer.CtrlDescriptor, 0); // Memory map AAXI Lite register block
    //Buffers are inputs, not from UIO
    DMAControlBuffer.BufDescriptor[0] = open("/dev/uio1", O_RDWR | O_SYNC); // Open /dev/uio1 which represents the dma source buffer
    DMAControlBuffer.BufMMAPPointer[0]  = mmap(NULL, 65535, PROT_READ | PROT_WRITE, MAP_SHARED, DMAControlBuffer.BufDescriptor[0], 0); // Memorymap source address
    DMAControlBuffer.BufDescriptor[1] = open("/dev/uio2", O_RDWR | O_SYNC); // Open /dev/uio2 which represents the dma destination buffer
    DMAControlBuffer.BufMMAPPointer[1] = mmap(NULL, 65535, PROT_READ | PROT_WRITE, MAP_SHARED, DMAControlBuffer.BufDescriptor[1], 0); // Memory

    return PyModule_Create(&DMALib_Module);
}
