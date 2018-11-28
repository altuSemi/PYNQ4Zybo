//////////////////////////////////////////////////////////////////////
//libdma.c      : DMA lib for pynq
//Purpose       : allocate memory , control , transfer and receive data to/from ZYNQ PL 
//Written by    : altuSemi
//Date          : Nov 20, 2018
/////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <sys/mman.h>
#include <string.h>
#include <stdint.h>
#include "libdma.h"

/*Allocate contiguous DRAM buffer*/
/*unsigned int * DRAMAlloc(unsigned int len) {
        unsigned int *buffer = malloc(len * sizeof(unsigned int));
        buffer[0]=1974;
        buffer[1]=9;
        buffer[2]=27;
        return buffer;
}*/


/* Create a function that can free our pointers */
void DRAMFreePtr(void *ptr)
{   
    free(ptr);
}

void DMASet(unsigned int* dma_virtual_address, unsigned int offset, unsigned int value) {
    dma_virtual_address[offset>>2] = value;
}

unsigned int DMAGet(unsigned int* dma_virtual_address, unsigned int offset) {
    return dma_virtual_address[offset>>2];
}

void DMAS2MMStatus(unsigned int* dma_virtual_address) {
    unsigned int status = DMAGet(dma_virtual_address, S2MM_STATUS_REGISTER);
    printf("Stream to memory-mapped status (0x%08x@0x%02x):", (unsigned int) status, S2MM_STATUS_REGISTER);
    if (status & 0x00000001) printf(" halted"); else printf(" running");
    if (status & 0x00000002) printf(" idle");
    if (status & 0x00000008) printf(" SGIncld");
    if (status & 0x00000010) printf(" DMAIntErr");
    if (status & 0x00000020) printf(" DMASlvErr");
    if (status & 0x00000040) printf(" DMADecErr");
    if (status & 0x00000100) printf(" SGIntErr");
    if (status & 0x00000200) printf(" SGSlvErr");
    if (status & 0x00000400) printf(" SGDecErr");
    if (status & 0x00001000) printf(" IOC_Irq");
    if (status & 0x00002000) printf(" Dly_Irq");
    if (status & 0x00004000) printf(" Err_Irq");
    printf("\n");
}

void DMAMM2SStatus(unsigned int* dma_virtual_address) {
    unsigned int status = DMAGet(dma_virtual_address, MM2S_STATUS_REGISTER);
    printf("Memory-mapped to stream status (0x%08x@0x%02x):", (unsigned int) status, MM2S_STATUS_REGISTER);
    if (status & 0x00000001) printf(" halted"); else printf(" running");
    if (status & 0x00000002) printf(" idle");
    if (status & 0x00000008) printf(" SGIncld");
    if (status & 0x00000010) printf(" DMAIntErr");
    if (status & 0x00000020) printf(" DMASlvErr");
    if (status & 0x00000040) printf(" DMADecErr");
    if (status & 0x00000100) printf(" SGIntErr");
    if (status & 0x00000200) printf(" SGSlvErr");
    if (status & 0x00000400) printf(" SGDecErr");
    if (status & 0x00001000) printf(" IOC_Irq");
    if (status & 0x00002000) printf(" Dly_Irq");
    if (status & 0x00004000) printf(" Err_Irq");
    printf("\n");
}



void DMAMM2SSync(unsigned int* dma_virtual_address) {
    unsigned int mm2s_status =  DMAGet(dma_virtual_address, MM2S_STATUS_REGISTER);
    while(!(mm2s_status & 1<<12) || !(mm2s_status & 1<<1) ){
        //DMAS2MMStatus(dma_virtual_address);
        //DMAMM2SStatus(dma_virtual_address);

        mm2s_status =  DMAGet(dma_virtual_address, MM2S_STATUS_REGISTER);
    }
}

void DMAS2MMSync(unsigned int* dma_virtual_address) {
    unsigned int s2mm_status = DMAGet(dma_virtual_address, S2MM_STATUS_REGISTER);
    while(!(s2mm_status & 1<<12) || !(s2mm_status & 1<<1)){
        //DMAS2MMStatus(dma_virtual_address);
        //DMAMM2SStatus(dma_virtual_address);

        s2mm_status = DMAGet(dma_virtual_address, S2MM_STATUS_REGISTER);
    }
}

void memdump(void* virtual_address, int byte_count) {
    char *p = virtual_address;
   int offset;
    for (offset = 0; offset < byte_count; offset++) {
        printf("%02x", p[offset]);
        if (offset % 4 == 3) { printf(" "); }
    }
    printf("\n");
}


void  DMATransfer () {
    //printf("Hello DMA\n");
    bool SrcBuf=DMAControlBuffer.BufSelect;
    int  len=DMAControlBuffer.BufLength;
    bool DstBuf=1-SrcBuf;
    
    //printf("Source memory block:      "); memdump(DMAControlBuffer.BufMMAPPointer[SrcBuf], 32);
    //printf("Destination memory block: "); memdump(DMAControlBuffer.BufMMAPPointer[DstBuf], 32);

    DMASet(DMAControlBuffer.CtrlMMAPPointer, S2MM_CONTROL_REGISTER, 4);
    DMASet(DMAControlBuffer.CtrlMMAPPointer, MM2S_CONTROL_REGISTER, 4);
    //DMAS2MMStatus(DMAControlBuffer.CtrlMMAPPointer);
    //DMAMM2SStatus(DMAControlBuffer.CtrlMMAPPointer);

    //printf("Halting DMA\n");
    DMASet(DMAControlBuffer.CtrlMMAPPointer, S2MM_CONTROL_REGISTER, 0);
    DMASet(DMAControlBuffer.CtrlMMAPPointer, MM2S_CONTROL_REGISTER, 0);
    //DMAS2MMStatus(DMAControlBuffer.CtrlMMAPPointer);
    //DMAMM2SStatus(DMAControlBuffer.CtrlMMAPPointer);

    //printf("Writing destination address\n");
    DMASet(DMAControlBuffer.CtrlMMAPPointer, S2MM_DESTINATION_ADDRESS, DMAControlBuffer.BufMemoryAddress[DstBuf]); // Write destination address
    //DMAS2MMStatus(DMAControlBuffer.CtrlMMAPPointer);

    //printf("Writing source address...\n");
    DMASet(DMAControlBuffer.CtrlMMAPPointer, MM2S_START_ADDRESS,DMAControlBuffer.BufMemoryAddress[SrcBuf]); // Write source address
    //DMAMM2SStatus(DMAControlBuffer.CtrlMMAPPointer);

    //printf("Starting S2MM channel with all interrupts masked...\n");
    DMASet(DMAControlBuffer.CtrlMMAPPointer, S2MM_CONTROL_REGISTER, 0xf001);
    //DMAS2MMStatus(DMAControlBuffer.CtrlMMAPPointer);

    //printf("Starting MM2S channel with all interrupts masked...\n");
    DMASet(DMAControlBuffer.CtrlMMAPPointer, MM2S_CONTROL_REGISTER, 0xf001);
    //DMAMM2SStatus(DMAControlBuffer.CtrlMMAPPointer);

    //printf("Writing S2MM transfer length...\n");
    DMASet(DMAControlBuffer.CtrlMMAPPointer, S2MM_LENGTH, len<<2);
    //DMAS2MMStatus(DMAControlBuffer.CtrlMMAPPointer);

    //printf("Writing MM2S transfer length...\n");
    DMASet(DMAControlBuffer.CtrlMMAPPointer, MM2S_LENGTH, len<<2);
    //DMAMM2SStatus(DMAControlBuffer.CtrlMMAPPointer);

    //printf("Waiting for MM2S synchronization...\n");
    DMAMM2SSync(DMAControlBuffer.CtrlMMAPPointer);

    //printf("Waiting for S2MM sychronization...\n");
    DMAS2MMSync(DMAControlBuffer.CtrlMMAPPointer); // If this locks up make sure all memory ranges are assigned under Address Editor!

    //DMAS2MMStatus(DMAControlBuffer.CtrlMMAPPointer);
    //DMAMM2SStatus(DMAControlBuffer.CtrlMMAPPointer);

    //printf("Destination memory block: "); memdump(DMAControlBuffer.BufMMAPPointer[DstBuf], 32);                         
    return;
}
