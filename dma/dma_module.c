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
			NULL,		//Control Pointer
 			CONTROL_PHY_ADD,//Control Physical Address
			0,		//Buf select 0/1
			0,		//Buf length
			0,		//Buffer0 Descriptor
			0,		//Buffer1 Descriptor
			NULL,		//Buffer0 Pointer
			NULL	, 	//Buffer1 Pointer
			BUFFER0_PHY_ADD,//Buffer0 Physical Address
			BUFFER1_PHY_ADD //Buffer1 Physical Address
};


/* Create a function that can free our pointers */
static PyObject * DMALib_DRAMFreeMem(PyObject *self, PyObject *args)
{
    munmap(DMAControlBuffer.CtrlMMAPPointer,65535);
    munmap(DMAControlBuffer.BufMMAPPointer[0],65535);
    munmap(DMAControlBuffer.BufMMAPPointer[1],65535);
    return Py_BuildValue("i", true);;
}

static PyObject * DMALib_DRAMMMAP(PyObject *self, PyObject *args)
{
    //DMA control and buffers struct init:
    DMAControlBuffer.CtrlDescriptor = open("/dev/uio0", O_RDWR | O_SYNC); // Open /dev/uio0 which represents the dma control area
    DMAControlBuffer.CtrlMMAPPointer = mmap(NULL, 65535, PROT_READ | PROT_WRITE, MAP_SHARED,DMAControlBuffer.CtrlDescriptor, 0); // Memory map AXI Lite register block
    //Buffers are inputs, not from UIO
    DMAControlBuffer.BufDescriptor[0] = open("/dev/uio1", O_RDWR | O_SYNC); // Open /dev/uio1 which represents the dma source buffer
    DMAControlBuffer.BufMMAPPointer[0]  = mmap(NULL, 65535, PROT_READ | PROT_WRITE, MAP_SHARED, DMAControlBuffer.BufDescriptor[0], 0); // Memory map source address
    DMAControlBuffer.BufDescriptor[1] = open("/dev/uio2", O_RDWR | O_SYNC); // Open /dev/uio2 which represents the dma destination buffer
    DMAControlBuffer.BufMMAPPointer[1] = mmap(NULL, 65535, PROT_READ | PROT_WRITE, MAP_SHARED, DMAControlBuffer.BufDescriptor[1], 0); // Memory
    return Py_BuildValue("i", true);;
}

static PyObject * DMALib_DMABufGet(PyObject *self, PyObject *args) {

  uint32_t * DstPointer=DMAControlBuffer.BufMMAPPointer[!DMAControlBuffer.BufSelect];
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
    if (n > 16000) {
        return NULL;
    }

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
}

static PyMethodDef DMALib_FunctionsTable[] = {  
      {
        "DRAMFreeMem",      // name exposed to Python
        DMALib_DRAMFreeMem, // C wrapper function
        METH_VARARGS,          // received variable args (but really just 1)
        "Free memory buffers" // documentation
    },{
        "DRAMMMAP",      // name exposed to Python
        DMALib_DRAMMMAP, // C wrapper function
        METH_VARARGS,          // received variable args (but really just 1)
        "Map memory buffers" // documentation
    },{
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
    DMAControlBuffer.CtrlMMAPPointer = mmap(NULL, 65535, PROT_READ | PROT_WRITE, MAP_SHARED,DMAControlBuffer.CtrlDescriptor, 0); // Memory map AXI Lite register block
    //Buffers are inputs, not from UIO
    DMAControlBuffer.BufDescriptor[0] = open("/dev/uio1", O_RDWR | O_SYNC); // Open /dev/uio1 which represents the dma source buffer
    DMAControlBuffer.BufMMAPPointer[0]  = mmap(NULL, 65535, PROT_READ | PROT_WRITE, MAP_SHARED, DMAControlBuffer.BufDescriptor[0], 0); // Memorymap source address
    DMAControlBuffer.BufDescriptor[1] = open("/dev/uio2", O_RDWR | O_SYNC); // Open /dev/uio2 which represents the dma destination buffer
    DMAControlBuffer.BufMMAPPointer[1] = mmap(NULL, 65535, PROT_READ | PROT_WRITE, MAP_SHARED, DMAControlBuffer.BufDescriptor[1], 0); // Memory

    return PyModule_Create(&DMALib_Module);
}
