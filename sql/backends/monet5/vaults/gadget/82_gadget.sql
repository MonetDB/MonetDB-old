create procedure gadgetAttach(fname string) external name gadget.attach;
create procedure gadgetLoad(tname string) external name gadget.load;
create procedure gadgetLoadAll(tname string) external name gadget.loadall;
create procedure gadgetListDir(dname string) external name gadget.listdir;
create procedure gadgetListDirPat(dname string, pattern string) external name gadget.listdirpattern;
create procedure gadgetListDirAll(dname string) external name gadget.listdirall;
create procedure gadgetAttachAll(*) external name gadget.attachall;
create procedure gadgetCheckTable(tname string) external name gadget.checktable;
create function gadgetPHkeyConvert(x real, y real, z real, baseSize real, nbits int) returns bigint external name gadget.phkeyconvert;
create function gadgetPHkeyInvert(phkey bigint, baseSize real, nbits int) returns table (x real, y real, z real, cSize real) external name gadget.phkeyinvert;

