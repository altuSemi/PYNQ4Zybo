#!/bin/bash
sudo busybox devmem 0x01010000 w 0x00000000
sudo busybox devmem 0x01010004 w 0x00000000
sudo busybox devmem 0x01010008 w 0x00000000
sudo busybox devmem 0x0101000C w 0x00000000
sudo busybox devmem 0x01010000
sudo busybox devmem 0x01010004
sudo busybox devmem 0x01010008
sudo busybox devmem 0x0101000C
sudo busybox devmem 0x01000000 w 0x11223344
sudo busybox devmem 0x01000004 w 0x11223344
sudo busybox devmem 0x01000008 w 0x11223344
sudo busybox devmem 0x0100000C w 0x11223344
sudo busybox devmem 0x40400004
sudo busybox devmem 0x40400034
sudo busybox devmem 0x40400000 w 0x4
sudo busybox devmem 0x40400030 w 0x4
sudo busybox devmem 0x40400034
sudo busybox devmem 0x40400004
sudo busybox devmem 0x40400018 w 0x01000000
sudo busybox devmem 0x40400048 w 0x01010000
sudo busybox devmem 0x40400000 w 0x0000f001
sudo busybox devmem 0x40400030 w 0x0000f001
sudo busybox devmem 0x40400004
sudo busybox devmem 0x40400034
sudo busybox devmem 0x40400028 w 0x10
sudo busybox devmem 0x40400058 w 0x10
sudo busybox devmem 0x40400004
sudo busybox devmem 0x40400034
sudo busybox devmem 0x01010000
sudo busybox devmem 0x01010004
sudo busybox devmem 0x01010008
sudo busybox devmem 0x0101000C
