# PYNQ4Zybo
Instructions and packages for Zybo compatibility to Pynq

Pynq repository was not targeted for the Zybo board. However, since the Zybo board has a Zynq device on it, Pynq can be ported unto it. The following steps were applied successfully on Pynq Z1 V2.0 image.


## Precompiled Image

The first step is to  <a href="https://files.digilent.com/Products/PYNQ/pynq_z1_v2.0.img.zip" target="_blank">download the precompiled image</a> and write the image to a micro SD card. This image is targeted for Pynq Z1 board. Its BOOT partition includes the following files:
BOOT.bin        -   Binary boot file, which includes the Zynq bitstream, FirstStageBootLoader and U-boot
uImage          -   Kernel image file
devicetree.dtb  -   device tree blob
These files were compiled for  the Pynq-Z1 and have to be replaced with files that are compiled for the Zybo board. The 2nd partition which includes the linux root file system (and the pynq package). This partition can remain as is, beside a package suggested below for dma access from python.

## Compiling u-boot and linux kernel for Zybo

The kernel files below are compiled on Ubuntu v16.04.1, installed on VM VirtualBox version 5.2.
The linux and u-boot repository version is <a href="https://www.xilinx.com/support/answers/68370.html" target="_blank">2016.4</a>.
Prior to compiling, the environment should be setup according to the steps recommended on the <a href="https://pynq.readthedocs.io/en/v2.0/pynq_sd_card.html" target="_blank">Pynq</a> help page:

  1. Install Vivado 2016.1 and Xilinx SDK 2016.1
  2. Install dependencies using the following script from <a href="https://github.com/Xilinx/PYNQ/tree/v2.0" target="_blank">PYNQ</a> repository:
```
   <PYNQ repository>/sdbuild/scripts/setup_host.sh
```
  3. Source the appropriate settings files from Vivado and Xilinx SDK - add to ~/.bashrc:
```
source /opt/Xilinx/SDK/2016.4/settings64.sh
source /opt/Xilinx/Vivado/2016.4/settings64.sh
export CROSS_COMPILE=arm-xilinx-linux-gnueabi-
```
The boot file compilation are based on steps 1 - 3 mentioned in this <a href="https://superuser.blog/pynq-linux-on-zedboard/" target="_blank">Zeb board</a>.   pynq porting guide., with these modifications:
### Step 1 -  Zybo u-boot make command:
```
make zynq_zybo_config
make
```
### Step 3 -  Zybo devicetree blob:
The file to be edited is <a href="https://github.com/altuSemi/PYNQ4Zybo/blob/master/zynq-zybo.dts" target="_blank">/linux-xlnx/arch/arm/boot/dts/zynq-zybo.dts</a>.
On top of that, the dma package is using 3 generic-uio which should be defined in <a href="https://github.com/altuSemi/PYNQ4Zybo/blob/master/zynq-7000.dtsi" target="_blank">/linux-xlnx/arch/arm/boot/dts/zynq-7000.dtsi</a>. The following should be added under 'amba':
```
amba: amba {
		u-boot,dm-pre-reloc;
		compatible = "simple-bus";
		#address-cells = <1>;
		#size-cells = <1>;
		interrupt-parent = <&intc>;
		ranges;
		axi_dma: axi-dma@40400000 {
			compatible = "generic-uio";
			interrupt-parent = <&intc>;
			interrupts = <0 29 4>;
			reg = <0x40400000 0x10000>;
		};
		dma_src@010000000 {
			compatible = "generic-uio";
			reg = <0x01000000 0x01000000>;
			interrupt-parent = <&intc>;
			interrupts = <0 30 4>;
		};

		dma_dst@020000000 {
			compatible = "generic-uio";
			reg = <0x02000000 0x01000000>;
			interrupt-parent = <&intc>;
			interrupts = <0 31 4>;
		};
```
## Booting PYNQ on Zybo
Next step is to copy the boot files over the original files in the Pynq sdcard BOOT partition. Then place the sd card in the Zybo sd card slot, set the boot jumper to boot from sd-card, and turn on the board.
The Zybo should boot and load the Linux kernel. The board can be accessed via UART, remote login or the Jupyter notebook portal as described in the <a href="https://pynq.readthedocs.io/en/v2.0/getting_started.html" target="_blank"> Pynq documentation page</a>.

## Overlays
This Pynq4Zybo porting guide was verified with two overlay designs:
### 1. <a href="https://pynq.readthedocs.io/en/v2.0/overlay_design_methodology/overlay_tutorial.html" target="_blank"> Adder overlay</a> - a simple overlay which implements an adder in the PL. 
Also explained in this <a href="https://www.youtube.com/watch?v=Dupyek4NUoI target="_blank"> video-guide</a>.
Jupyter notebook can be found here:
### 2. <a href="https://www.youtube.com/watch?v=LoLCtSzj9BU" target="_blank"> Function acceleration with Zynq</a> - Low pass filter acceleration with PL logic and AXI dma.
The contiguous memory allocation and dma access failed to work following this porting guide.
Instead a python c extension was designed to pass data from python to the PL and back via the AXI dma. This extension is currently supporting a max buffer length of 16K word (unit32).
The package is installed by copying the <a href="https://github.com/altuSemi/PYNQ4Zybo/tree/master/dma" target="_blank">dma</a> directory to the board (vias the samba server), and executing insidet he dma directory:
```
pip3.6 instyall . 
```
The code of the dma package is based on the following references:


Returning a numphy array from c: 		http://acooke.org/cute/ExampleCod0.html
Enhancing Python with Custom C Extensions:	https://stackabuse.com/enhancing-python-with-custom-c-extensions/
Lauri's Blog - AXI Direct Memory Access : 	https://lauri.xn--vsandi-pxa.com/hdl/zynq/xilinx-dma.html
generic-uio was used instead of devmem to overcome non-root permissions issue.

Enjoy!


