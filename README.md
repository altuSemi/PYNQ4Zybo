# PYNQ4Zybo
Instructions and packages for Zybo compatibility to Pynq

Pynq repository was not tagrgetted for the Zybo board. However, since the Zybo board has a Zynq device on it, Pynq can be ported unto it. The following steps were applied successfully on Pynq Z1 V2.0 image.


## Precompiled Image

The first step is to  <a href="https://files.digilent.com/Products/PYNQ/pynq_z1_v2.0.img.zip" target="_blank">download the precompiled image</a> and write the image to a micro SD card. This image is targeted for Pynq Z1 board. Its BOOT partition includes the following files:
BOOT.bin        -   Binary boot file, which includes the Zynq bitstream, FirstStageBootLoader and U-boot
uImage          -   Kernel image file
devicetree.dtb  -   device tree blob
These files were compiled for  the Pynq-Z1 and have to be replaced with files that are compiled for the Zybo board. The 2nd partition which includes the linux root file system (and the pynq pacakge). This partition can remain as is, beside a package suggested below for dma access from python.

## Compiling u-boot and linux kernel for Zybo

The kernel files below are compiled on Ubuntu v16.04.1, installed on VM VirtalBox version 5.2.
The linux and u-boot repository version is <a href="https://www.xilinx.com/support/answers/68370.html" target="_blank">2016.4 .
Prior to compiling, the enviorment should be setup according to the steps recommened on the <a href="https://pynq.readthedocs.io/en/v2.0/pynq_sd_card.html" target="_blank">Pynq help page:

  1. Install Vivado 2016.1 and Xilinx SDK 2016.1
  2. Install dependencies using the following script from <a href="https://github.com/Xilinx/PYNQ/tree/v2.0" target="_blank">PYNQ repository:
```
   <PYNQ repository>/sdbuild/scripts/setup_host.sh

\
  3. Source the appropriate settings files from Vivado and Xilinx SDK - add to ~/.bashrc:
  
```
source /opt/Xilinx/SDK/2016.4/settings64.sh
source /opt/Xilinx/Vivado/2016.4/settings64.sh
export CROSS_COMPILE=arm-xilinx-linux-gnueabi-
```

The following steps are based on a similar pynq porting method tagregetd for the <a href="https://superuser.blog/pynq-linux-on-zedboard/" target="_blank">Zeb board:
  
