---
title: xv6文件系统的扩容
date: 2019-09-28 18:07:54
categories:
- OS
tags:
- OS
---

## 准备措施
&emsp;&emsp;根据指导书里的内容，将文件`Makefile`下的 CPUS 改为1并在`QEMUOPTS`前添加语句`QEMUEXTRA $=-$snapshot`以加快xv6创建大文件的速率
```c
ifndef CPUS
CPUS := 1
endif
QEMUEXTRA = -snapshot
``` 
&emsp;&emsp;`mkfs.c`文件在初始化文件系统时通过参数`FSSIZE`指定分配数据块的数量
```c
int nbitmap = FSSIZE/(BSIZE*8) + 1;
```
&emsp;&emsp;所以我们需要将预定义在头文件`param.h`里的`FSSIZE`从1000修改为20000
```c
#define FSSIZE       20000  // size of file system in blocks
```
&emsp;&emsp;下载`big.c`并将big命令添加至`Makefile`下的UPROGS列表；在xv6的shell里运行big观察结果；目前xv6的文件系统可分配140个数据块
```c
UPROGS=\
	_cat\
	_echo\
	......\
	_zombie\
	_big\
```
```markdown
$ big
.
wrote 140 sectors
done; ok
```
## fs.c/fs.h相关项解读
&emsp;&emsp;inode定义于头文件`fs.h`内，初始的文件系统含有13个直接指针`NDIRECT`和1个一级间接指针`NINDIRECT`，`addrs`数组的最后一项就是间接块的地址；参数`MAXFILE`表征该文件系统所能指向的数据块数目的最大值，当前为12 + 512 / 4 = 140
```c
#define NDIRECT 12
#define NINDIRECT (BSIZE / sizeof(uint))
#define MAXFILE (NDIRECT + NINDIRECT)

// On-disk inode structure
struct dinode {
  short type;           // File type
  short major;          // Major device number (T_DEV only)
  short minor;          // Minor device number (T_DEV only)
  short nlink;          // Number of links to inode in file system
  uint size;            // Size of file (bytes)
  uint addrs[NDIRECT + 1];   // Data block addresses
};
```
&emsp;&emsp;函数`bmap(ip, bn)`负责建立inode与磁盘上存储的数据块间的映射，返回inode`ip`中的第`bn`个数据块号；如果没有这样一个数据块，`bmap`就会分配一个；未申请的块用块号0表示，当`bmap`遇到0时就将它们替换为新的块号
```c
static uint
bmap(struct inode *ip, uint bn)
{
  uint addr, *a;
  struct buf *bp;

  if(bn < NDIRECT){
    // 块号为零，需要替换
    if((addr = ip->addrs[bn]) == 0)
      ip->addrs[bn] = addr = balloc(ip->dev);
    return addr;
  }
  bn -= NDIRECT; //无符号数，方便下面比大小

  if(bn < NINDIRECT){
    // Load indirect block, allocating if necessary.
    // ip->addrs[NDIRECT]指向第一个间接块
    if((addr = ip->addrs[NDIRECT]) == 0)
      ip->addrs[NDIRECT] = addr = balloc(ip->dev);
    bp = bread(ip->dev, addr);
    a = (uint*)bp->data;
    if((addr = a[bn]) == 0){
      a[bn] = addr = balloc(ip->dev);
      log_write(bp);
    }
    brelse(bp);
    return addr;
  }

  panic("bmap: out of range");
}
```
&emsp;&emsp;我们的主要改动在于`fs.h`指针的重新分配，然后对`bmap`做相应的修改即可

## 相关修改
&emsp;&emsp;首先改动`fs.h`内的指针分配，把一个直接指针替换为二级间接指针`DNINDIRECT`，此时文件系统有11个直接指针，一个一级间接指针和一个二级间接指针，`addrs`数组的倒数第二项是第一个间接块的地址，最后一项为第二个间接块的地址。此时`MAXFILE`的值为11 + 512 / 4 + (512 / 4)$^2$ = 16523
```c
#define NDIRECT 11
#define NINDIRECT (BSIZE / sizeof(uint))
#define DINDIRECT NINDIRECT * NINDIRECT
#define MAXFILE (NDIRECT + NINDIRECT + DINDIRECT)

// On-disk inode structure
struct dinode {
  short type;           // File type
  short major;          // Major device number (T_DEV only)
  short minor;          // Minor device number (T_DEV only)
  short nlink;          // Number of links to inode in file system
  uint size;            // Size of file (bytes)
  uint addrs[NDIRECT + 2];   // Data block addresses
};
```
&emsp;&emsp;然后为`fs.c`添加二级间接指针的映射关系，这里唯一需要注意的就是**如何计算二级间接指针的逻辑坐标**

![os](https://raw.githubusercontent.com/plumprc/plumprc.github.io/master/_posts/xv6/material/os.png)
&emsp;&emsp;如图，记`bn`为逻辑上我们需要的数据块号（比如5000），我们需要知道该块对应的一级间接指针坐标以及逻辑上的二级间接指针坐标，只需要进行如下简单的运算：
* 一级间接指针坐标：bn / 128
* 二级间接指针坐标：bn % 128

&emsp;&emsp;以5000为例，经计算得到bn位于第39个一级间接指针指向的指针块，在该块上的下标为8。补充完整的`bmap`代码如下：
```c
static uint
bmap(struct inode *ip, uint bn)
{
  uint addr, *a, *p1, p1_addr, *p2, p2_addr;
  struct buf *bp, *bp2;

  if(bn < NDIRECT){
    if((addr = ip->addrs[bn]) == 0)
      ip->addrs[bn] = addr = balloc(ip->dev);
    return addr;
  }
  bn -= NDIRECT; //无符号数，方便下面比大小，且逻辑坐标归零

  if(bn < NINDIRECT){
    // Load indirect block, allocating if necessary.
    // ip->addrs[NDIRECT]指向第一个间接块
    if((addr = ip->addrs[NDIRECT]) == 0)
      ip->addrs[NDIRECT] = addr = balloc(ip->dev);
    bp = bread(ip->dev, addr);
    a = (uint*)bp->data;
    if((addr = a[bn]) == 0){
      a[bn] = addr = balloc(ip->dev);
      log_write(bp);
    }

    brelse(bp);
    return addr;
  }
  bn -= NINDIRECT;

  if(bn < DINDIRECT){
    // Load double indirect block, allocating if necessary
    // ip->addrs[NDIRECT + 1]指向第二个间接块
    if((addr = ip->addrs[NDIRECT + 1]) == 0)
      ip->addrs[NDIRECT + 1] = addr = balloc(ip->dev);
    bp = bread(ip->dev, addr); // 读第二个间接块指向的指针块
    p1 = (uint*)bp->data;
    p1_addr = bn / NINDIRECT; // 计算bn对应的指针块的位置，取值范围为(0, 127)
    if((addr = p1[p1_addr]) == 0){
      p1[p1_addr] = addr = balloc(ip->dev);
      log_write(bp);
    }

    bp2 = bread(ip->dev, addr);
    p2 = (uint*)bp2->data;
    p2_addr = bn % NINDIRECT; // 计算指针块上bn的下标
    if((addr = p2[p2_addr]) == 0){
      p2[p2_addr] = addr = balloc(ip->dev);
      log_write(bp2);
    }

    brelse(bp2);
    brelse(bp);
    return addr;
  }

  panic("bmap: out of range");
}
```
&emsp;&emsp;完成修改后重新执行`make qemu`，键入`big`命令得到以下结果，可见文件系统成功得到扩充，现在最大可指向16523个数据块，大小约为8.46MB
```markdown
$ big
..................................................
..................................................
..................................................
...............
wrote 16523 sectors
done; ok
```