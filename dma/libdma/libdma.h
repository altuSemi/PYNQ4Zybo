//////////////////////////////////////////////////////////////////////
//libdma.c      : DMA lib for pynq
//Purpose       : allocate memory , control , transfer and receive data to/from ZYNQ PL 
//Written by    : altuSemi
//Date          : Nov 20, 2018
/////////////////////////////////////////////////////////////////////

#define MM2S_CONTROL_REGISTER 0x00
#define MM2S_STATUS_REGISTER 0x04
#define MM2S_START_ADDRESS 0x18
#define MM2S_LENGTH 0x28

#define S2MM_CONTROL_REGISTER 0x30
#define S2MM_STATUS_REGISTER 0x34
#define S2MM_DESTINATION_ADDRESS 0x48
#define S2MM_LENGTH 0x58

/*Allocate contiguous DRAM buffer*/
//unsigned int * DRAMAlloc(unsigned int len);

/* Create a function that can free our pointers */
void DRAMFreePtr(void *ptr);

typedef enum { false, true } bool;

typedef struct DMACtrlBufStruct{

   int 			CtrlDescriptor;
   uint32_t *		CtrlMMAPPointer;
   const uint32_t 	CtrlMemoryAddress;//0x04040000;
   bool			BufSelect;
   int                  BufLength;
   int 			BufDescriptor[2];
   uint32_t *		BufMMAPPointer[2];
   const uint32_t 	BufMemoryAddress[2];//{0x01010000,0x01000000};
} DMACtrlBuf;


extern DMACtrlBuf DMAControlBuffer;

void DMASet(unsigned int* dma_virtual_address, unsigned int offset, unsigned int value) ;

unsigned int DMAGet(unsigned int* dma_virtual_address, unsigned int offset) ;

void DMAS2MMStatus(unsigned int* dma_virtual_address) ;

void DMAMM2SStatus(unsigned int* dma_virtual_address) ;

void DMAMM2SSync(unsigned int* dma_virtual_address) ;

void DMAS2MMSync(unsigned int* dma_virtual_address) ;

void memdump(void* virtual_address, int byte_count) ;

void  DMATransfer () ;
